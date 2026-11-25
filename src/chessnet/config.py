"""Configuration models, loaded from YAML files (see ``configs/``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FilterConfig(BaseModel):
    """Which games are kept for training.

    The result must be ``1-0``, ``0-1`` or ``1/2-1/2``, and the game must start from the
    standard position; neither is configurable. Games are assumed to be rated, because the
    source is a rated-games dump; the Event header is not checked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_elo: int = Field(default=2000, ge=0, description="minimum rating of both players")
    min_base_time: int = Field(default=180, ge=0, description="minimum base time, seconds")
    terminations: tuple[str, ...] = ("Normal", "Time forfeit")
    min_plies: int = Field(default=20, ge=1)


class DataConfig(BaseModel):
    """Preprocessing: filters, dataset size and shard layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filters: FilterConfig = FilterConfig()
    max_games: int | None = Field(default=None, ge=1, description="stop after this many kept")
    shard_size: int = Field(default=1_000_000, ge=1, description="positions per shard")
    num_buckets: int = Field(default=64, ge=1, description="shuffle buckets (train split)")
    seed: int = 0


class ModelConfig(BaseModel):
    """Network size: residual blocks x filters, squeeze-excitation and value-head widths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocks: int = Field(ge=1, description="residual blocks in the trunk")
    filters: int = Field(ge=1, description="channels of the trunk")
    se_ratio: int | None = Field(default=4, ge=1, description="SE reduction; null disables SE")
    value_channels: int = Field(default=32, ge=1)
    value_hidden: int = Field(default=128, ge=1)


#: The model presets (also in ``configs/model/``): tiny for tests and CI, base for Kaggle.
MODEL_PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(blocks=4, filters=64),
    "small": ModelConfig(blocks=6, filters=128),
    "base": ModelConfig(blocks=10, filters=128),
}


class TrainConfig(BaseModel):
    """A training run: model, batch, optimisation, schedule, logging and checkpoint cadence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(default="base", description="a MODEL_PRESETS name")
    batch_size: int = Field(default=1024, ge=1)
    epochs: float = Field(default=3.0, gt=0, description="passes over the train split")
    max_steps: int | None = Field(default=None, ge=1, description="caps the run; overrides epochs")
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    momentum: float = Field(default=0.9, ge=0, lt=1, description="SGD (Nesterov) only")
    warmup_steps: int = Field(default=1000, ge=0)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1, description="cosine floor, x lr")
    value_loss_weight: float = Field(default=0.5, ge=0)
    label_smoothing: float = Field(default=0.05, ge=0, lt=1)
    grad_clip: float = Field(default=5.0, gt=0, description="max gradient norm")
    amp: bool = Field(default=True, description="fp16 autocast + GradScaler (CUDA only)")
    compile: bool = Field(default=False, description="torch.compile the training graph")
    channels_last: bool = Field(
        default=False, description="NHWC memory format (can speed up fp16 convolutions)"
    )
    log_every: int = Field(default=100, ge=1, description="steps per progress line")
    val_every: int = Field(default=2000, ge=1, description="steps between validations")
    val_batches: int = Field(default=50, ge=1, description="size of the fixed val subset")
    checkpoint_minutes: float = Field(default=20.0, gt=0)
    keep_checkpoints: int = Field(default=3, ge=1)
    num_workers: int = Field(default=2, ge=0, description="DataLoader worker processes")
    seed: int = 0
    deterministic: bool = False

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        if value not in MODEL_PRESETS:
            raise ValueError(f"unknown model {value!r}; presets: {', '.join(MODEL_PRESETS)}")
        return value


def _read_yaml(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data_config(path: str | Path) -> DataConfig:
    """Read a :class:`DataConfig` from a YAML file."""
    return DataConfig.model_validate(_read_yaml(path) or {})


def load_model_config(path: str | Path) -> ModelConfig:
    """Read a :class:`ModelConfig` from a YAML file."""
    return ModelConfig.model_validate(_read_yaml(path))


def load_train_config(path: str | Path) -> TrainConfig:
    """Read a :class:`TrainConfig` from a YAML file."""
    return TrainConfig.model_validate(_read_yaml(path) or {})
