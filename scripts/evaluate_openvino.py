"""Phase 8: evaluate an OpenVINO-exported policy in the simulator.

Loads an export from scripts/export_openvino.py, wraps the compiled model behind LeRobot's
policy interface (`reset`, `select_action`; executes `n_action_steps` actions of each
predicted chunk, as LeRobot's ACT does), uses the checkpoint's own pre/post-processors
(normalisation), and runs the expert's evaluation loop (scripts/evaluate_expert.py) — so the result
compares directly with the PyTorch policy's `lerobot-eval` and with the expert's.

Usage:
    uv run python scripts/evaluate_openvino.py --export outputs/openvino/bc_mlp_last --seed 1000 --n-episodes 50
Output: outputs/eval/<task>/openvino_<export name>/eval_info.json (+ videos)
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim.lerobot_plugin  # noqa: E402,F401  (registers the waste_sort env with LeRobot)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_expert import evaluate, make_vec_env  # noqa: E402  (the same loop as the expert)
from sim.lerobot_plugin import WasteSortEnvConfig  # noqa: E402
from sim.waste_env import TASKS  # noqa: E402


class OpenVINOPolicy(nn.Module):
    """A compiled OpenVINO model behind LeRobot's policy interface. Inputs arrive already
    normalised by the checkpoint's preprocessor; the action leaves normalised and is
    de-normalised by its postprocessor."""

    def __init__(self, compiled, input_keys: list[str], n_action_steps: int):
        super().__init__()
        self.compiled, self.input_keys, self.n_action_steps = compiled, input_keys, n_action_steps
        self.queue: deque = deque()

    def reset(self):
        self.queue.clear()

    def select_action(self, batch):
        if not self.queue:
            chunk = self.compiled([batch[k].cpu().numpy() for k in self.input_keys])[0]  # (B, chunk, D)
            self.queue.extend(torch.from_numpy(np.ascontiguousarray(chunk[:, i]))
                              for i in range(min(self.n_action_steps, chunk.shape[1])))
        return self.queue.popleft()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True, help="folder written by export_openvino.py")
    ap.add_argument("--task", default="t0_can_to_metal_bin", choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, required=True, help="first seed; episodes use seed, seed+1, ...")
    ap.add_argument("--n-episodes", type=int, required=True)
    ap.add_argument("--device", default=None, help="OpenVINO device (default: the one exported for)")
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import openvino as ov
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins

    info = json.loads((args.export / "export_info.json").read_text())
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(info["checkpoint"])
    pre, post = make_pre_post_processors(cfg, pretrained_path=info["checkpoint"],
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    run_cfg = {"INFERENCE_PRECISION_HINT": info["inference_precision"]}  # same precision as checked at export
    compiled = ov.Core().compile_model(args.export / "model.xml", args.device or info["device"], run_cfg)
    policy = OpenVINOPolicy(compiled, info["inputs"], info["n_action_steps"])

    # the environment renders the cameras the policy was trained on, at its image size
    image_keys = info["inputs"][1:]
    _, h, w = info["input_shapes"][image_keys[0]][1:]
    env_cfg = WasteSortEnvConfig(task=args.task, cameras=[k.rsplit(".", 1)[1] for k in image_keys],
                                 observation_height=h, observation_width=w)
    vec = make_vec_env(env_cfg)
    evaluate(vec, env_cfg, policy, pre, post, args.seed, args.n_episodes,
             args.out or Path("outputs/eval") / args.task / f"openvino_{args.export.name}",
             f"OpenVINO {info['policy_type']} ({args.device or info['device']})", args.videos)
    vec.close()


if __name__ == "__main__":
    main()
