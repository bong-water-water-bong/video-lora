"""Scheduler configuration registry for video generation models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SchedulerConfig:
    """Configuration for a diffusion scheduler."""

    name: str
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    timestep_spacing: str = "leading"
    steps_offset: int = 1

    # Model-specific overrides
    rescale_betas_zero_snr: bool = False


# Common presets
SCHEDULER_PRESETS: dict[str, SchedulerConfig] = {
    "sd15": SchedulerConfig(name="PNDM", num_train_timesteps=1000),
    "sdxl": SchedulerConfig(name="DDPM", num_train_timesteps=1000),
    "wan": SchedulerConfig(
        name="FlowMatchEulerDiscrete",
        num_train_timesteps=1000,
        prediction_type="flow_matching",
        timestep_spacing="trailing",
    ),
    "ltx": SchedulerConfig(
        name="FlowMatchEulerDiscrete",
        num_train_timesteps=1000,
        prediction_type="flow_matching",
        timestep_spacing="trailing",
    ),
    "cogvideo": SchedulerConfig(
        name="DDIM",
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    ),
}


def get_scheduler_config(model_type: str) -> SchedulerConfig:
    """Get the scheduler config for a model type."""
    if model_type not in SCHEDULER_PRESETS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(SCHEDULER_PRESETS.keys())}"
        )
    return SCHEDULER_PRESETS[model_type]
