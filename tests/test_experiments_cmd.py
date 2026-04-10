"""Tests for the `engram experiments list` command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engram.cli import app

runner = CliRunner()


def _seed_index(root: Path, entries: list[dict]) -> None:
    """Write a project with a populated experiments.jsonl."""
    (root / 'engram.yaml').write_text('name: test\n')
    (root / 'experiments').mkdir()
    index_path = root / 'experiments' / 'experiments.jsonl'
    index_path.write_text('\n'.join(json.dumps(e) for e in entries) + '\n')


def _make_entry(  # noqa: PLR0913 — test helper; every parameter is a meaningful index field
    exp_id: str,
    impl: str,
    dataset: str,
    timestamp: str,
    accuracy: float,
    f1: float,
    cost: float,
    matched: int = 10,
) -> dict:
    return {
        'id': exp_id,
        'implementation': impl,
        'dataset': dataset,
        'timestamp': timestamp,
        'models': ['test-model'],
        'matched_examples': matched,
        'macro_accuracy': accuracy,
        'macro_f1': f1,
        'field_accuracy': {'topic': accuracy},
        'field_precision': {'topic': accuracy},
        'field_recall': {'topic': accuracy},
        'field_f1': {'topic': f1},
        'cost': {'total_usd': cost, 'avg_usd': cost / 10},
    }


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An empty index prints a friendly hint, not a crash."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')
    (tmp_path / 'experiments').mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'No experiments in the index yet' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_shows_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Populated index renders a table with the expected columns and values."""
    _seed_index(
        tmp_path,
        [
            _make_entry('exp-a', 'classify-anthropic', 'sample', '2026-04-01T12:00:00Z', 0.95, 0.92, 0.05),
            _make_entry('exp-b', 'classify-openai', 'sample', '2026-04-02T12:00:00Z', 0.88, 0.85, 0.02),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'exp-a' in result.output
    assert 'exp-b' in result.output
    assert 'classify-anthropic' in result.output
    assert 'classify-openai' in result.output
    # Metric cells rendered as percentages.
    assert '95.0%' in result.output
    assert '92.0%' in result.output
    # Cost formatted as dollars.
    assert '$0.05' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_sorted_newest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Newer experiments appear before older ones in the rendered output."""
    _seed_index(
        tmp_path,
        [
            _make_entry('old-exp', 'classify-anthropic', 'sample', '2026-04-01T08:00:00Z', 0.9, 0.9, 0.01),
            _make_entry('new-exp', 'classify-anthropic', 'sample', '2026-04-10T08:00:00Z', 0.9, 0.9, 0.01),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    # new-exp should appear first in the text output
    assert result.output.index('new-exp') < result.output.index('old-exp')


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_filter_by_impl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_index(
        tmp_path,
        [
            _make_entry('ant', 'classify-anthropic', 'sample', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01),
            _make_entry('oai', 'classify-openai', 'sample', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list', '--impl', 'classify-openai'])
    assert result.exit_code == 0
    assert 'oai' in result.output
    assert 'ant' not in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_filter_by_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_index(
        tmp_path,
        [
            _make_entry('a', 'classify-anthropic', 'sample-a', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01),
            _make_entry('b', 'classify-anthropic', 'sample-b', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list', '-d', 'sample-b'])
    assert result.exit_code == 0
    assert 'b' in result.output
    # Use a unique ID fragment so filtering is unambiguous across Rich wrapping.
    assert 'sample-a' not in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_filter_with_no_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_index(
        tmp_path,
        [_make_entry('a', 'classify-anthropic', 'sample', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01)],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list', '--impl', 'ghost-impl'])
    assert result.exit_code == 0
    assert 'No experiments match the given filters' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_limit_truncates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entries = [
        _make_entry(f'exp-{i}', 'classify-anthropic', 'sample', f'2026-04-{i:02d}T12:00:00Z', 0.9, 0.9, 0.01)
        for i in range(1, 6)
    ]
    _seed_index(tmp_path, entries)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list', '--limit', '2'])
    assert result.exit_code == 0
    # Only the two newest should appear.
    assert 'exp-5' in result.output
    assert 'exp-4' in result.output
    assert 'exp-1' not in result.output
    # Truncation hint tells the user how many there are in total.
    assert 'Showing 2 of 5 experiments' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_limit_zero_means_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entries = [
        _make_entry(f'exp-{i}', 'classify-anthropic', 'sample', f'2026-04-{i:02d}T12:00:00Z', 0.9, 0.9, 0.01)
        for i in range(1, 6)
    ]
    _seed_index(tmp_path, entries)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list', '--limit', '0'])
    assert result.exit_code == 0
    for i in range(1, 6):
        assert f'exp-{i}' in result.output


def test_experiments_list_json_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With --json, output is parseable JSON with the filtered entries."""
    _seed_index(
        tmp_path,
        [
            _make_entry('exp-a', 'classify-anthropic', 'sample', '2026-04-01T12:00:00Z', 0.95, 0.92, 0.05),
            _make_entry('exp-b', 'classify-openai', 'sample', '2026-04-02T12:00:00Z', 0.88, 0.85, 0.02),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['--json', 'experiments', 'list'])
    assert result.exit_code == 0

    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    # Still sorted newest-first.
    assert parsed[0]['id'] == 'exp-b'
    assert parsed[1]['id'] == 'exp-a'
    assert parsed[0]['macro_f1'] == 0.85


def test_experiments_list_no_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Outside a project, the command exits 1 with a clear message."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 1
    assert 'No engram.yaml found' in result.output
