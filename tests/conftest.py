"""Shared test fixtures."""

import os
from pathlib import Path

import pytest

from engram.observability.output_mode import reset_output_mode


@pytest.fixture(autouse=True)
def _reset_output_mode():
    """Reset output mode between tests."""
    reset_output_mode()
    yield
    reset_output_mode()


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
