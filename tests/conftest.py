"""Shared test fixtures."""

import os
from pathlib import Path

import pytest

from engram.observability.output_mode import OutputMode, reset_output_mode


@pytest.fixture(autouse=True)
def _reset_output_mode():
    """Reset output mode between tests."""
    reset_output_mode()
    yield
    reset_output_mode()


@pytest.fixture
def rich_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Rich output mode with a wide enough virtual terminal so Rich tables render in full."""

    # CliRunner captures stdout, so sys.stdout.isatty() is False and
    # OutputMode.detect() picks JSON by default. Override detect, and set COLUMNS
    # so Rich stops truncating cells to the 80-char fallback.
    def _detect(force_json: bool = False) -> OutputMode:  # noqa: ARG001 — match detect signature
        return OutputMode(use_rich=True, use_json_logging=False)

    monkeypatch.setattr('engram.observability.output_mode.OutputMode.detect', _detect)
    monkeypatch.setenv('COLUMNS', '200')


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal engram project directory."""
    (tmp_path / 'engram.yaml').write_text('name: test-project\ndescription: test\n')
    for d in ['workflows', 'implementations', 'datasets', 'experiments']:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def _chdir_project(project_dir: Path):
    """Change to a project directory for the duration of a test."""
    original = Path.cwd()
    os.chdir(project_dir)
    yield project_dir
    os.chdir(original)
