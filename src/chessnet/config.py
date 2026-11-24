"""Configuration models, loaded from YAML files (see ``configs/``)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FilterConfig(BaseModel):
    """Which games are kept for training.

    The result must be ``1-0``, ``0-1`` or ``1/2-1/2``, and the game must start from the
    standard position; neither is configurable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rated_only: bool = True
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


def load_data_config(path: str | Path) -> DataConfig:
    """Read a :class:`DataConfig` from a YAML file."""
    with Path(path).open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return DataConfig.model_validate(raw or {})
