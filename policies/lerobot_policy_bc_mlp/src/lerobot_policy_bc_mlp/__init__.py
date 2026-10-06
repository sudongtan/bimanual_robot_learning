"""Behaviour cloning baseline for LeRobot: `--policy.type=bc_mlp`.

Importing this package registers the policy type (the config's `register_subclass`);
LeRobot finds the policy class in `modeling_bc_mlp` and the processors in
`processor_bc_mlp` by naming convention (lerobot/policies/factory.py).
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed; this policy plugin needs it.") from e

from .configuration_bc_mlp import BCMLPConfig
from .modeling_bc_mlp import BCMLPPolicy
from .processor_bc_mlp import make_bc_mlp_pre_post_processors

__all__ = ["BCMLPConfig", "BCMLPPolicy", "make_bc_mlp_pre_post_processors"]
