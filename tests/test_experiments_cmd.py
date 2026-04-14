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
    # Full ids are hidden from rich output — only #N + impl/dataset + metrics remain.
    assert 'exp-a' not in result.output
    assert 'exp-b' not in result.output
    assert 'classify-anthropic' in result.output
    assert 'classify-openai' in result.output
    # Short ids shown.
    assert '1' in result.output
    assert '2' in result.output
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
    # Newer experiment (by timestamp) appears first — compare via the When column contents.
    assert result.output.index('2026-04-10') < result.output.index('2026-04-01')


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
    assert 'classify-openai' in result.output
    assert 'classify-anthropic' not in result.output


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
    # Only the two newest should appear — check via When column.
    assert '2026-04-05' in result.output
    assert '2026-04-04' in result.output
    assert '2026-04-01' not in result.output
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
    # All five experiments appear — check via their When column dates.
    for i in range(1, 6):
        assert f'2026-04-{i:02d}' in result.output


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


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_label_column_shown_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When at least one entry has a label, the Label column appears for all rows."""
    _seed_index(
        tmp_path,
        [
            _make_entry('a', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01),
            _make_entry('b', 'impl', 'sample', '2026-04-01T12:00:00Z', 0.8, 0.8, 0.02),
        ],
    )
    index_path = tmp_path / 'experiments' / 'experiments.jsonl'
    lines = index_path.read_text().strip().splitlines()
    entry_a = json.loads(lines[0])
    entry_a['label'] = 'prompt-v2'
    index_path.write_text(json.dumps(entry_a) + '\n' + lines[1] + '\n')
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'Label' in result.output
    assert 'prompt-v2' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_label_column_hidden_when_no_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When no entry has a label, the Label column is not shown at all."""
    _seed_index(
        tmp_path,
        [_make_entry('a', 'impl', 'sample', '2026-04-01T12:00:00Z', 0.9, 0.9, 0.01)],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'Label' not in result.output


def test_experiments_list_no_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Outside a project, the command exits 1 with a clear message."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 1
    assert 'No engram.yaml found' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_shows_short_id_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The `#` column renders the chronological short id computed at display time."""
    _seed_index(
        tmp_path,
        [
            _make_entry('newer', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01),
            _make_entry('older', 'impl', 'sample', '2026-04-01T12:00:00Z', 0.8, 0.8, 0.02),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    # Ascending chronological numbering — `older` is #1, `newer` is #2 — rendered as bare ints in the `#` column.
    assert '1' in result.output
    assert '2' in result.output
    # Full ids are not shown in rich mode.
    assert 'newer' not in result.output
    assert 'older' not in result.output


def _write_dataset_labels(root: Path, dataset: str, labels: dict) -> str:
    """Write a dataset's labels.json and return its content hash."""
    from engram.datasets.loader import compute_labels_hash  # noqa: PLC0415

    ds_dir = root / 'datasets' / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / 'labels.json').write_text(json.dumps(labels))
    return compute_labels_hash(labels)


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_marks_stale_rows_and_prints_footer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A row whose stored labels_hash no longer matches the dataset gets a yellow asterisk and a footer hint."""
    stored_labels = {'001.txt': {'topic': 'A'}}
    _write_dataset_labels(tmp_path, 'sample', stored_labels)
    # The index row was scored against a different labels payload.
    stale_entry = _make_entry('exp-a', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01)
    stale_entry['labels_hash'] = 'a' * 64  # any hash that won't match current labels

    _seed_index(tmp_path, [stale_entry])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert '*' in result.output
    assert 'labels have changed since this score' in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_does_not_mark_when_hash_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A row whose stored labels_hash matches the current dataset hash has no marker."""
    labels = {'001.txt': {'topic': 'A'}}
    current_hash = _write_dataset_labels(tmp_path, 'sample', labels)
    entry = _make_entry('exp-a', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01)
    entry['labels_hash'] = current_hash

    _seed_index(tmp_path, [entry])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'labels have changed since this score' not in result.output


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_skips_marker_without_stored_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rows with no labels_hash (older scores) don't get the stale marker even if labels exist on disk."""
    _write_dataset_labels(tmp_path, 'sample', {'001.txt': {'topic': 'A'}})
    entry = _make_entry('exp-a', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01)

    _seed_index(tmp_path, [entry])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'labels have changed since this score' not in result.output


def test_experiments_list_json_surfaces_labels_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """JSON mode tags stale rows with labels_stale=True."""
    _write_dataset_labels(tmp_path, 'sample', {'001.txt': {'topic': 'A'}})
    stale = _make_entry('exp-a', 'impl', 'sample', '2026-04-02T12:00:00Z', 0.9, 0.9, 0.01)
    stale['labels_hash'] = 'a' * 64

    _seed_index(tmp_path, [stale])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['--json', 'experiments', 'list'])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed[0].get('labels_stale') is True


@pytest.mark.usefixtures('rich_mode')
def test_experiments_list_rich_hides_full_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rich mode never shows the long underscore-joined full id, even for long names."""
    _seed_index(
        tmp_path,
        [
            _make_entry(
                'classify-anthropic_sample_20260412_235937',
                'classify-anthropic',
                'sample',
                '2026-04-12T23:59:37Z',
                0.95,
                0.92,
                0.05,
            ),
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['experiments', 'list'])
    assert result.exit_code == 0
    assert 'classify-anthropic_sample_20260412_235937' not in result.output
    assert '1' in result.output
    assert 'classify-anthropic' in result.output
