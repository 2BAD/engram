"""Smoke tests for the engram CLI."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engram.cli import app
from engram.config.validation import validate_project

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'engram' in result.output.lower()


def test_init_creates_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 0

    # Directory skeleton
    assert (tmp_path / 'engram.yaml').exists()
    assert (tmp_path / '.gitignore').exists()
    assert (tmp_path / 'workflows').is_dir()
    assert (tmp_path / 'implementations').is_dir()
    assert (tmp_path / 'datasets').is_dir()
    assert (tmp_path / 'experiments' / '.gitignore').exists()

    # Example workflow, implementation, and dataset
    assert (tmp_path / 'workflows' / 'classify' / 'workflow.yaml').exists()
    assert (tmp_path / 'implementations' / 'classify-api' / 'implementation.yaml').exists()
    assert (tmp_path / 'implementations' / 'classify-api' / 'prompts' / 'system.md').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'dataset.yaml').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '001.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '002.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '003.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'labels.json').exists()

    # Quickstart instructions are printed
    assert 'engram eval classify-api sample' in result.output


def test_init_project_passes_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The scaffolded project loads through the full config pipeline with zero errors.

    This exercises load_workflow + load_implementation (including the runner and
    scorer name validators) and the cross-reference check in validate_project.
    """
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 0

    errors = validate_project(tmp_path)
    assert errors == [], f'scaffolded project has validation errors: {errors}'


def test_init_labels_match_workflow_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The scaffolded labels.json uses only field values that exist in the workflow enum."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ['init'])

    labels = json.loads((tmp_path / 'datasets' / 'sample' / 'labels.json').read_text())
    valid_topics = {'billing', 'technical', 'feedback'}
    valid_sentiments = {'positive', 'negative', 'neutral'}
    for filename, label in labels.items():
        assert label['topic'] in valid_topics, f'{filename}: bad topic {label["topic"]}'
        assert label['sentiment'] in valid_sentiments, f'{filename}: bad sentiment {label["sentiment"]}'


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
