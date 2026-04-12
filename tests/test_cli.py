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

    # Shared workflow and dataset
    assert (tmp_path / 'workflows' / 'classify' / 'workflow.yaml').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'dataset.yaml').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '001.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '002.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'inputs' / '003.txt').exists()
    assert (tmp_path / 'datasets' / 'sample' / 'labels.json').exists()

    # Both implementations scaffolded so the compare flow is runnable out of the box
    assert (tmp_path / 'implementations' / 'classify-anthropic' / 'implementation.yaml').exists()
    assert (tmp_path / 'implementations' / 'classify-anthropic' / 'prompts' / 'system.md').exists()
    assert (tmp_path / 'implementations' / 'classify-openai' / 'implementation.yaml').exists()
    assert (tmp_path / 'implementations' / 'classify-openai' / 'prompts' / 'system.md').exists()

    # Quickstart instructions mention both implementations and the compare step
    assert 'engram run classify-anthropic --dataset sample' in result.output
    assert 'engram run classify-openai --dataset sample' in result.output
    assert 'engram compare' in result.output


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


def test_init_scaffolds_env_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """init writes a .env.example listing the two API keys used by the scaffolded runners."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 0

    env_example = (tmp_path / '.env.example').read_text()
    assert 'ANTHROPIC_API_KEY=' in env_example
    assert 'OPENAI_API_KEY=' in env_example
    # And the quickstart message tells the user how to activate it.
    assert 'cp .env.example .env' in result.output


def test_cli_loads_dotenv_from_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Running any engram command in a project populates os.environ from <root>/.env."""
    # Minimal project skeleton (no workflows needed — status tolerates an empty project).
    (tmp_path / 'engram.yaml').write_text('name: envtest\n')
    for d in ['workflows', 'implementations', 'datasets', 'experiments']:
        (tmp_path / d).mkdir()
    (tmp_path / '.env').write_text('ENGRAM_TEST_KEY_FROM_DOTENV=hello-from-env\n')

    monkeypatch.delenv('ENGRAM_TEST_KEY_FROM_DOTENV', raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['status'])
    assert result.exit_code == 0
    # The callback ran and set the var before the subcommand executed.
    assert os.environ.get('ENGRAM_TEST_KEY_FROM_DOTENV') == 'hello-from-env'


def test_cli_outside_project_does_not_crash_without_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Running engram outside a project (no engram.yaml anywhere above) is a silent no-op for .env loading."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0


def test_run_command_prints_next_step_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """After a successful run, the command prints the experiment ID and the exact next command."""
    monkeypatch.chdir(tmp_path)

    # Scaffold a real project, satisfy the preflight, then stub run_eval.
    init_result = runner.invoke(app, ['init'])
    assert init_result.exit_code == 0
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    monkeypatch.setattr(
        'engram.cli.commands.run.run_eval',
        lambda *_args, **_kwargs: ('classify-anthropic_sample_fake-id', 7),
    )

    result = runner.invoke(app, ['run', 'classify-anthropic', '--dataset', 'sample'])

    assert result.exit_code == 0
    assert 'Experiment complete' in result.output
    # Output shows the short_id and impl/dataset. Full id is hidden from users.
    assert '#7' in result.output
    assert 'classify-anthropic/sample' in result.output
    assert 'classify-anthropic_sample_fake-id' not in result.output
    # Hint block names the score command with the short_id and the list command.
    assert 'engram score 7 --save' in result.output
    assert 'engram experiments list' in result.output


def test_run_command_with_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """engram run --label echoes the label in the completion message."""
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ['init'])
    assert init_result.exit_code == 0
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-test')
    monkeypatch.setattr(
        'engram.cli.commands.run.run_eval',
        lambda *_args, **_kwargs: ('fake-id', 8),
    )

    result = runner.invoke(app, ['run', 'classify-anthropic', '--dataset', 'sample', '--label', 'prompt-v2'])

    assert result.exit_code == 0
    assert '#8' in result.output
    assert 'prompt-v2' in result.output


def test_run_command_preflight_rejects_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """engram run fails fast with a friendly message when the runner's API key env var is missing."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.chdir(tmp_path)

    # Scaffold a project via init so we have a valid implementation to point at.
    init_result = runner.invoke(app, ['init'])
    assert init_result.exit_code == 0

    # No .env file and no exported key → preflight should catch it before any API call.
    result = runner.invoke(app, ['run', 'classify-anthropic', '--dataset', 'sample'])

    assert result.exit_code == 1
    assert 'ANTHROPIC_API_KEY' in result.output
    assert 'Missing required environment variable' in result.output
    # The error message tells the user where to put the key.
    assert '.env' in result.output
    assert 'export ANTHROPIC_API_KEY' in result.output


def test_status_no_project(tmp_path: Path):
    os.chdir(tmp_path)
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 1


@pytest.mark.usefixtures('_chdir_project')
def test_status_empty_project():
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 0
    assert 'test-project' in result.output
