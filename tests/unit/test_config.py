from pathlib import Path

import pytest
from pydantic import ValidationError

from chessnet.config import DataConfig, FilterConfig, load_data_config

REPO = Path(__file__).parents[2]


def test_defaults_match_the_documented_filters() -> None:
    config = DataConfig()
    assert config.filters == FilterConfig(
        rated_only=True,
        min_elo=2000,
        min_base_time=180,
        terminations=("Normal", "Time forfeit"),
        min_plies=20,
    )
    assert config.max_games is None


def test_repository_config_loads() -> None:
    config = load_data_config(REPO / "configs" / "data" / "lichess_2000plus.yaml")
    assert config.filters.min_elo == 2000
    assert config.filters.min_base_time == 180
    assert config.filters.terminations == ("Normal", "Time forfeit")
    assert config.max_games == 1_000_000
    assert config.shard_size == 1_000_000


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_data_config(path) == DataConfig()


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text("filters:\n  minimum_elo: 1800\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="minimum_elo"):
        load_data_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [{"shard_size": 0}, {"num_buckets": 0}, {"max_games": 0}],
)
def test_invalid_values_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        DataConfig.model_validate(kwargs)


def test_negative_rating_floor_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FilterConfig(min_elo=-1)
