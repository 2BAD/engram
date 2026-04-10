"""Tests for baseline tracking and the baseline CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engram.cli import app
from engram.tracking.baseline import (
    get_impl_reference,
    get_workflow_baseline,
    load_baselines,
    lookup_experiment,
    set_impl_reference,
    set_workflow_baseline,
)

from .test_tracking import _setup_project_with_experiments

runner = CliRunner(env={'COLUMNS': '200'})


# --- Module-level helpers ---


def test_load_baselines_empty(tmp_path: Path):
    (tmp_path / 'experiments').mkdir()
    assert load_baselines(tmp_path) == {}


def test_set_and_get_workflow_baseline(tmp_path: Path):
    id_a, _id_b = _setup_project_with_experiments(tmp_path)
    set_workflow_baseline(tmp_path, 'classify', id_a)

    assert get_workflow_baseline(tmp_path, 'classify') == id_a
    assert get_workflow_baseline(tmp_path, 'missing') is None


def test_set_and_get_impl_reference(tmp_path: Path):
    id_a, _id_b = _setup_project_with_experiments(tmp_path)
    set_impl_reference(tmp_path, 'classify', 'classify-api', id_a)

    assert get_impl_reference(tmp_path, 'classify', 'classify-api') == id_a
    assert get_impl_reference(tmp_path, 'classify', 'other') is None
    assert get_impl_reference(tmp_path, 'other-workflow', 'classify-api') is None


def test_baseline_and_reference_coexist(tmp_path: Path):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    set_workflow_baseline(tmp_path, 'classify', id_a)
    set_impl_reference(tmp_path, 'classify', 'classify-api', id_b)

    data = load_baselines(tmp_path)
    assert data['classify']['baseline'] == id_a
    assert data['classify']['references']['classify-api'] == id_b


def test_lookup_experiment(tmp_path: Path):
    id_a, _id_b = _setup_project_with_experiments(tmp_path)
    workflow, impl = lookup_experiment(tmp_path, id_a)
    assert workflow == 'classify'
    assert impl == 'classify-api'


def test_lookup_experiment_missing(tmp_path: Path):
    (tmp_path / 'experiments').mkdir()
    with pytest.raises(FileNotFoundError):
        lookup_experiment(tmp_path, 'does-not-exist')


def test_save_baselines_writes_pretty_json(tmp_path: Path):
    (tmp_path / 'experiments').mkdir()
    set_workflow_baseline(tmp_path, 'classify', 'exp-1')

    text = (tmp_path / 'experiments' / 'baselines.json').read_text()
    # Pretty-printed (multi-line) and parseable
    assert '\n' in text
    assert json.loads(text) == {'classify': {'baseline': 'exp-1'}}


# --- CLI smoke tests ---


def test_baseline_set_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, _id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['baseline', 'set', id_a])
    assert result.exit_code == 0
    assert get_workflow_baseline(tmp_path, 'classify') == id_a


def test_baseline_promote_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, _id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['baseline', 'promote', id_a])
    assert result.exit_code == 0
    assert get_impl_reference(tmp_path, 'classify', 'classify-api') == id_a


def test_baseline_set_cli_rejects_missing_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['baseline', 'set', 'bogus-id'])
    assert result.exit_code == 1
    assert load_baselines(tmp_path) == {}


@pytest.mark.usefixtures('rich_mode')
def test_baseline_show_cli_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['baseline', 'show'])
    assert result.exit_code == 0
    assert 'No baselines set' in result.output


def test_baseline_show_cli_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    set_workflow_baseline(tmp_path, 'classify', id_a)
    set_impl_reference(tmp_path, 'classify', 'classify-api', id_b)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['baseline', 'show'])
    assert result.exit_code == 0
    assert 'classify' in result.output
    assert id_a in result.output
    assert id_b in result.output


# --- Compare fallback ---


def test_compare_uses_workflow_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    set_workflow_baseline(tmp_path, 'classify', id_a)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['compare', id_b])
    assert result.exit_code == 0
    # Both IDs appear in the rendered table headers
    assert id_a in result.output
    assert id_b in result.output


def test_compare_single_arg_without_baseline_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['compare', id_b])
    assert result.exit_code == 1
    assert 'baseline' in result.output.lower()


def test_compare_two_args_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['compare', id_a, id_b])
    assert result.exit_code == 0
    assert id_a in result.output
    assert id_b in result.output


def test_compare_against_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['compare', id_b, '--against', id_a])
    assert result.exit_code == 0
    assert id_a in result.output
    assert id_b in result.output


# --- Init scaffold ---


def test_init_gitignore_includes_baselines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['init'])
    assert result.exit_code == 0

    gitignore = (tmp_path / 'experiments' / '.gitignore').read_text()
    assert '!baselines.json' in gitignore
