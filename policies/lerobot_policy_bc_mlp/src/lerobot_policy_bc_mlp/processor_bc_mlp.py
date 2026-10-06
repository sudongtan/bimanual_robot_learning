from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_bc_mlp import BCMLPConfig


def make_bc_mlp_pre_post_processors(
    config: BCMLPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """LeRobot's default pipelines, as ACT uses (policies/act/processor_act.py): normalise inputs
    with the dataset statistics (modes from `config.normalization_mapping`), add the batch
    dimension, move to the device; de-normalise the action and move it back to the CPU."""
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
