from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("bc_mlp")
@dataclass
class BCMLPConfig(PreTrainedConfig):
    """Single-step behaviour cloning: current camera images + joint state -> the next action.

    The baseline the ACT paper calls BC-ConvMLP (Zhao et al. 2023, arXiv 2304.13705): no action
    chunking, no history, no latent variable. Image encoder, normalisation and L1 loss are the same
    as LeRobot's ACT, so a comparison with ACT isolates what action chunking (and its CVAE) adds.
    """

    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image encoder: one ResNet shared by all cameras, ImageNet weights (as ACT), global-average
    # pooled to one feature vector per camera.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    # MLP over [camera features..., joint state].
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512])
    dropout: float = 0.1

    # Training preset (AdamW, as ACT; backbone at a lower rate).
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a torchvision ResNet. Got {self.vision_backbone}.")
        if self.n_obs_steps != 1:
            raise ValueError("bc_mlp uses only the current observation (n_obs_steps=1).")

    def validate_features(self) -> None:
        if not self.image_features and self.robot_state_feature is None:
            raise ValueError("bc_mlp needs at least one camera image or the robot state as input.")
        if self.action_feature is None:
            raise ValueError("bc_mlp needs 'action' among the output features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]  # only the action at the current step

    @property
    def reward_delta_indices(self) -> None:
        return None
