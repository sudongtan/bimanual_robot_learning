"""The BC baseline policy plugin (policies/lerobot_policy_bc_mlp): LeRobot finds it by name, and a
training step and an action step run with the dataset's feature shapes."""

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERAS = ("overview", "left_wrist", "right_wrist")


def features(h=64, w=64):
    inputs = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
              **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in CAMERAS}}
    return inputs, {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))}


def test_lerobot_discovers_the_plugin():
    """Installed as `lerobot_policy_bc_mlp`, LeRobot imports it at startup and resolves the
    policy class and processors by naming convention."""
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    assert get_policy_class("bc_mlp").__name__ == "BCMLPPolicy"
    assert type(make_policy_config("bc_mlp")).__name__ == "BCMLPConfig"


def test_train_step_and_action():
    from lerobot_policy_bc_mlp import BCMLPConfig, BCMLPPolicy

    inputs, outputs = features()
    cfg = BCMLPConfig(input_features=inputs, output_features=outputs, device="cpu",
                      pretrained_backbone_weights=None)   # no download in tests
    policy = BCMLPPolicy(cfg)
    b = 4
    batch = {OBS_STATE: torch.randn(b, 12), ACTION: torch.randn(b, 1, 12),
             "action_is_pad": torch.zeros(b, 1, dtype=torch.bool),
             **{f"observation.images.{c}": torch.rand(b, 3, 64, 64) for c in CAMERAS}}
    policy.train()
    loss, info = policy.forward(batch)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss) and "l1_loss" in info
    assert all(p.grad is not None for p in policy.mlp.parameters())

    policy.reset()
    action = policy.select_action({k: v for k, v in batch.items() if k.startswith("observation")})
    assert action.shape == (b, 12)
    assert policy.predict_action_chunk(batch).shape == (b, 1, 12)

    # padded steps (past an episode's end) contribute no loss
    batch["action_is_pad"][:] = True
    assert policy.forward(batch)[0].item() == 0.0


def test_openvino_export_matches_pytorch(tmp_path):
    """Phase 8: the exported network gives the PyTorch policy's action chunk (f32 inference;
    the CPU default on ARM is f16, debug-log #33)."""
    import importlib.util

    import numpy as np
    import openvino as ov
    from lerobot_policy_bc_mlp import BCMLPConfig, BCMLPPolicy

    spec = importlib.util.spec_from_file_location("export_openvino", "scripts/export_openvino.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    inputs, outputs = features(32, 32)
    cfg = BCMLPConfig(input_features=inputs, output_features=outputs, device="cpu", pretrained_backbone_weights=None)
    policy = BCMLPPolicy(cfg).eval()
    state_key, image_keys, example = exp.example_inputs(cfg)
    net = exp.ChunkNet(policy, state_key, image_keys).eval()
    with torch.no_grad():
        ref = net(*example).numpy()
        ov_model = ov.convert_model(net, example_input=tuple(example))
    ov.save_model(ov_model, tmp_path / "model.xml")
    compiled = ov.Core().compile_model(tmp_path / "model.xml", "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    got = compiled([x.numpy() for x in example])[0]
    assert got.shape == ref.shape == (1, 1, 12)
    assert np.abs(got - ref).max() < 1e-4
