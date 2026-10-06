"""Phase 8: convert a LeRobot policy checkpoint to OpenVINO, check it, and time it.

What is converted: the policy network only — normalised joint state + normalised camera
images in, a chunk of normalised actions out (`policy.predict_action_chunk`). Normalising
inputs and de-normalising actions stay in LeRobot's pre/post-processors, loaded from the
same checkpoint at run time (scripts/evaluate_openvino.py).

Conversion uses OpenVINO's PyTorch route (`ov.convert_model(module, example_input=...)`,
`ov.save_model`; docs.openvino.ai, "Converting PyTorch Models"). It traces the network, so
it suits single-pass policies (bc_mlp, ACT); SmolVLA (tokeniser + iterative sampling) is not
handled here.

Checks written to export_info.json:
  - parity: max |OpenVINO − PyTorch| on random inputs of the right shapes, at f32 (is the
    conversion correct?) and at the precision OpenVINO runs at (the device default unless
    --precision is given — f16 on this ARM Mac's CPU, debug-log #33);
  - latency: mean/median ms per call, OpenVINO (on --device) vs PyTorch on the CPU.

Usage:
    uv run python scripts/export_openvino.py --checkpoint outputs/train/t0_can_to_metal_bin/bc_mlp/checkpoints/last/pretrained_model
Output: outputs/openvino/<policy type>_<checkpoint step>/ (or --out): model.xml/.bin + export_info.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ChunkNet(nn.Module):
    """The policy network with positional inputs (state, image_1, ..., image_n), as OpenVINO
    needs them; returns the normalised action chunk (B, chunk, action_dim)."""

    def __init__(self, policy, state_key: str, image_keys: list[str]):
        super().__init__()
        self.policy, self.state_key, self.image_keys = policy, state_key, image_keys

    def forward(self, state, *images):
        batch = {self.state_key: state, **dict(zip(self.image_keys, images))}
        return self.policy.predict_action_chunk(batch)


def load_policy(checkpoint: Path):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()  # our bc_mlp plugin registers itself on import
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.eval()
    return cfg, policy


def example_inputs(cfg, batch: int = 1) -> tuple[str, list[str], list[torch.Tensor]]:
    from lerobot.utils.constants import OBS_STATE

    image_keys = list(cfg.image_features)
    state = torch.randn(batch, *cfg.input_features[OBS_STATE].shape)
    images = [torch.randn(batch, *cfg.input_features[k].shape) for k in image_keys]
    return OBS_STATE, image_keys, [state, *images]


def time_calls(fn, n: int) -> dict:
    for _ in range(5):  # warm-up
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t))
    return {"mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="a lerobot-train pretrained_model folder")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, NPU or AUTO")
    ap.add_argument("--fp16", action="store_true", help="store weights in FP16 (ov.save_model compress_to_fp16)")
    ap.add_argument("--precision", default=None, help="INFERENCE_PRECISION_HINT, e.g. f32, f16, bf16 (default: device default)")
    ap.add_argument("--n-timing", type=int, default=50)
    args = ap.parse_args()

    import openvino as ov

    cfg, policy = load_policy(args.checkpoint)
    state_key, image_keys, inputs = example_inputs(cfg)
    net = ChunkNet(policy, state_key, image_keys).eval()
    out = args.out or Path("outputs/openvino") / f"{cfg.type}_{args.checkpoint.parent.name}"
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        ov_model = ov.convert_model(net, example_input=tuple(inputs))
    ov.save_model(ov_model, out / "model.xml", compress_to_fp16=args.fp16)
    core = ov.Core()
    run_cfg = {"INFERENCE_PRECISION_HINT": args.precision} if args.precision else {}
    compiled = core.compile_model(out / "model.xml", args.device, run_cfg)
    precision = compiled.get_property("INFERENCE_PRECISION_HINT").to_string()  # e.g. "f16"

    with torch.no_grad():
        ref = net(*inputs).numpy()
    np_inputs = [x.numpy() for x in inputs]
    parity = float(np.abs(compiled(np_inputs)[0] - ref).max())
    f32 = core.compile_model(out / "model.xml", args.device, {"INFERENCE_PRECISION_HINT": "f32"})
    parity_f32 = float(np.abs(f32(np_inputs)[0] - ref).max())

    with torch.no_grad():
        torch_ms = time_calls(lambda: net(*inputs), args.n_timing)
    ov_ms = time_calls(lambda: compiled(np_inputs), args.n_timing)

    info = {
        "checkpoint": str(args.checkpoint.resolve()), "policy_type": cfg.type,
        "inputs": [state_key, *image_keys],
        "input_shapes": {k: list(x.shape) for k, x in zip([state_key, *image_keys], inputs)},
        "output": "normalised action chunk (batch, chunk, action_dim)", "output_shape": list(ref.shape),
        "n_action_steps": int(getattr(cfg, "n_action_steps", ref.shape[1])),
        "device": args.device, "fp16_weights": args.fp16, "inference_precision": precision,
        "parity_max_abs_diff": {"f32": parity_f32, precision: parity},
        "latency": {"openvino": ov_ms, "torch_cpu": torch_ms},
        "openvino_version": ov.__version__,
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    print(f"exported {cfg.type} -> {out / 'model.xml'}")
    print(f"parity: max |OpenVINO - PyTorch| = {parity_f32:.2e} at f32, {parity:.2e} at {precision} (the run precision)")
    print(f"latency per call: OpenVINO {args.device} ({precision}) {ov_ms['median_ms']:.1f} ms, "
          f"PyTorch CPU {torch_ms['median_ms']:.1f} ms")


if __name__ == "__main__":
    main()
