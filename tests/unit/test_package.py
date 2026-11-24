import re

from typer.testing import CliRunner

import chessnet
from chessnet.cli import app

runner = CliRunner()


def test_version_is_pep440_dev_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", chessnet.__version__)


def test_cli_help_lists_program() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_without_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"chessnet {chessnet.__version__}"
