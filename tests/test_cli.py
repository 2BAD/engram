"""Smoke tests for the engram CLI."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engram.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'engram' in result.output.lower()


def test_init_creates_project(tmp_path: Path):
    os.chdir(tmp_path)
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 0
    assert (tmp_path / 'engram.yaml').exists()
    assert (tmp_path / 'workflows').is_dir()
    assert (tmp_path / 'implementations').is_dir()
    assert (tmp_path / 'datasets').is_dir()
    assert (tmp_path / 'experiments').is_dir()
    assert (tmp_path / 'experiments' / '.gitignore').exists()


def test_init_refuses_existing(tmp_path: Path):
    os.chdir(tmp_path)
    (tmp_path / 'engram.yaml').write_text('name: existing\n')
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 1


def test_status_no_project(tmp_path: Path):
    os.chdir(tmp_path)
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 1


@pytest.mark.usefixtures('_chdir_project')
def test_status_empty_project():
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 0
    assert 'test-project' in result.output
