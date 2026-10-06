"""Single-step behaviour cloning: ResNet image features + joint state -> MLP -> next action.

    each camera image ─ ResNet-18 (shared, ImageNet weights) ─ global avg pool ─ 512 ─┐
    joint state (12) ──────────────────────────────────────────────────────────────────┼─ concat ─ MLP ─ action (12)

Trained by regressing the demonstrated action (L1 loss, as LeRobot's ACT). At run time it is
called every control step and returns one action.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bc_mlp import BCMLPConfig


class BCMLPPolicy(PreTrainedPolicy):
    config_class = BCMLPConfig
    name = "bc_mlp"

    def __init__(self, config: BCMLPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        in_dim = 0
        if config.image_features:
            # Frozen batch-norm statistics, as ACT: training batches are small and correlated.
            resnet = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d
            )
            feat_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()  # keep the global-average-pooled features
            self.backbone = resnet
            in_dim += feat_dim * len(config.image_features)
        if config.robot_state_feature is not None:
            in_dim += config.robot_state_feature.shape[0]

        layers = []
        for h in config.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, config.action_feature.shape[0]))
        self.mlp = nn.Sequential(*layers)

    def get_optim_params(self) -> list[dict]:
        backbone = [p for n, p in self.named_parameters() if n.startswith("backbone.") and p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        return [{"params": rest}, {"params": backbone, "lr": self.config.optimizer_lr_backbone}]

    def reset(self):
        """No per-episode state: the policy sees only the current observation."""

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, action_dim) normalised action for the current observation."""
        feats = [self.backbone(batch[key]) for key in self.config.image_features]
        if self.config.robot_state_feature is not None:
            feats.append(batch[OBS_STATE])
        return self.mlp(torch.cat(feats, dim=-1))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """A chunk of one action: (B, 1, action_dim)."""
        self.eval()
        return self._predict(batch).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """The action for this control step: (B, action_dim)."""
        self.eval()
        return self._predict(batch)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """L1 loss between the predicted and the demonstrated action."""
        target = batch[ACTION][:, 0]                       # action_delta_indices = [0] -> (B, 1, D)
        err = F.l1_loss(self._predict(batch), target, reduction="none").mean(-1)
        if "action_is_pad" in batch:                       # padded past an episode's end: no target
            valid = ~batch["action_is_pad"][:, 0]
            err = err * valid
            loss = err.sum() / valid.sum().clamp_min(1) if reduction == "mean" else err
        else:
            loss = err.mean() if reduction == "mean" else err
        return loss, {"l1_loss": float(loss.mean().item())}
