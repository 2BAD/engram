"""Tests for the interactive experiment picker helper and its wiring into commands."""

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from engram.cli import app
from engram.cli.picker import pick_experiment_id

runner = CliRunner()


def _seed_index(root: Path, entries: list[dict]) -> None:
    (root / 'engram.yaml').write_text('name: test\n')
    (root / 'experiments').mkdir(exist_ok=True)
    (root / 'experiments' / 'experiments.jsonl').write_text('\n'.join(json.dumps(e) for e in entries) + '\n')


def _make_entry(exp_id: str, timestamp: str, accuracy: float = 1.0) -> dict:
    return {
        'id': exp_id,
        'implementation': 'classify-anthropic',
        'dataset': 'sample',
        'timestamp': timestamp,
        'matched_examples': 3,
        'macro_accuracy': accuracy,
        'macro_f1': accuracy,
    }


# --- Picker helper (unit) ---


def test_picker_exits_when_stdin_not_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Non-interactive stdin → friendly error + exit 1 instead of hanging on input()."""
    _seed_index(tmp_path, [_make_entry('only-exp', '2026-04-10T00:00:00')])
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: False)

    with pytest.raises(typer.Exit) as excinfo:
        pick_experiment_id(tmp_path)
    assert excinfo.value.exit_code == 1


def test_picker_exits_when_index_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Empty index → helpful error pointing at `engram run` + `engram score --save`."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')
    (tmp_path / 'experiments').mkdir()
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: True)

    with pytest.raises(typer.Exit) as excinfo:
        pick_experiment_id(tmp_path)
    assert excinfo.value.exit_code == 1


def test_picker_returns_selected_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Interactive mode: mock IntPrompt.ask and verify the right entry comes back."""
    _seed_index(
        tmp_path,
        [
            _make_entry('newest', '2026-04-10T02:00:00', accuracy=0.95),
            _make_entry('middle', '2026-04-09T12:00:00', accuracy=0.80),
            _make_entry('oldest', '2026-04-08T06:00:00', accuracy=0.60),
        ],
    )
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: True)
    monkeypatch.setattr('engram.cli.picker.IntPrompt.ask', lambda *_a, **_k: 2)

    assert pick_experiment_id(tmp_path) == 'middle'


def test_picker_sorts_newest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Picking choice 1 returns the most recent experiment, regardless of file order."""
    _seed_index(
        tmp_path,
        [
            _make_entry('mid', '2026-04-09T12:00:00'),
            _make_entry('old', '2026-04-08T06:00:00'),
            _make_entry('new', '2026-04-10T02:00:00'),
        ],
    )
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: True)
    monkeypatch.setattr('engram.cli.picker.IntPrompt.ask', lambda *_a, **_k: 1)

    assert pick_experiment_id(tmp_path) == 'new'


def test_picker_caps_to_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With more entries than `limit`, only the newest `limit` are offered."""
    entries = [_make_entry(f'exp-{i:02d}', f'2026-04-{i:02d}T00:00:00') for i in range(1, 15)]
    _seed_index(tmp_path, entries)
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: True)

    # Choice 1 should be exp-14 (newest), choice 10 should be exp-05 (5th newest).
    captured: dict = {}

    def _ask(*_args, choices: list[str], **_kwargs) -> int:  # type: ignore[no-untyped-def]
        captured['choices'] = choices
        return 1

    monkeypatch.setattr('engram.cli.picker.IntPrompt.ask', _ask)

    result = pick_experiment_id(tmp_path, limit=10)
    assert result == 'exp-14'
    assert len(captured['choices']) == 10


# --- End-to-end: commands dispatch through the picker when the ID is omitted ---


def test_score_command_picks_when_no_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`engram score` (no args) prompts the picker and scores the selected experiment."""
    from engram.scoring.engine import score_experiment  # noqa: PLC0415

    _setup_minimal_scored_project(tmp_path, 'exp-a')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('engram.cli.picker._is_interactive', lambda: True)
    monkeypatch.setattr('engram.cli.picker.IntPrompt.ask', lambda *_a, **_k: 1)

    # Sanity check the experiment is scorable before we run the command.
    assert score_experiment(tmp_path, 'exp-a').experiment_id == 'exp-a'

    result = runner.invoke(app, ['score'])
    assert result.exit_code == 0
    # The picker surfaced the ID and the score command ran against it.
    assert 'exp-a' in result.output


def test_score_command_non_tty_exits_with_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`engram score` (no args) in a non-interactive shell exits 1 with a helpful hint."""
    _setup_minimal_scored_project(tmp_path, 'exp-a')
    monkeypatch.chdir(tmp_path)
    # CliRunner stdin is not a TTY by default; no monkeypatch needed.

    result = runner.invoke(app, ['score'])
    assert result.exit_code == 1
    assert 'not interactive' in result.output
    assert 'pass the experiment ID' in result.output


def _setup_minimal_scored_project(root: Path, exp_id: str, short_id: int = 1) -> None:
    """Write a project with one workflow, one impl, one dataset, and one runnable experiment."""
    (root / 'engram.yaml').write_text('name: test\n')
    wf = root / 'workflows' / 'classify'
    wf.mkdir(parents=True)
    (wf / 'workflow.yaml').write_text(
        'name: classify\n'
        'output:\n'
        '  fields:\n'
        '    topic:\n'
        '      type: enum\n'
        '      values: [A, B]\n'
        'scorers:\n'
        '  topic: exact_match\n'
    )
    impl = root / 'implementations' / 'classify-api'
    impl.mkdir(parents=True)
    (impl / 'implementation.yaml').write_text('workflow: classify\nplatform: api\nrunner: anthropic\n')
    ds = root / 'datasets' / 'sample'
    ds.mkdir(parents=True)
    (ds / 'dataset.yaml').write_text('name: sample\n')
    (ds / 'labels.json').write_text(json.dumps({'001.txt': {'topic': 'A'}}))

    exp_dir = root / 'experiments' / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': short_id,
                'implementation': 'classify-api',
                'dataset': 'sample',
                'timestamp': '2026-04-10T00:00:00',
                'total': 1,
                'succeeded': 1,
                'failed': 0,
                'results': [
                    {
                        'input_file': '001.txt',
                        'output': {'topic': 'A'},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                        'cost_usd': 0.01,
                        'latency_ms': 100,
                        'error': '',
                    }
                ],
            }
        )
    )
    # Seed the index so the picker has something to show.
    (root / 'experiments' / 'experiments.jsonl').write_text(
        json.dumps(
            {
                'short_id': short_id,
                'id': exp_id,
                'implementation': 'classify-api',
                'dataset': 'sample',
                'timestamp': '2026-04-10T00:00:00',
                'models': ['claude-sonnet'],
                'matched_examples': 1,
                'macro_accuracy': 1.0,
                'macro_f1': 1.0,
            }
        )
        + '\n'
    )


def test_score_command_accepts_short_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`engram score 1` resolves the short_id through the index and scores the matching experiment."""
    _setup_minimal_scored_project(tmp_path, 'exp-a', short_id=1)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['score', '1'])
    assert result.exit_code == 0
    assert 'exp-a' in result.output


def test_score_command_rejects_unknown_short_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unknown short_id exits with a clear 'No experiment found' message."""
    _setup_minimal_scored_project(tmp_path, 'exp-a', short_id=1)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ['score', '99'])
    assert result.exit_code == 1
    assert 'No experiment found with short_id #99' in result.output
