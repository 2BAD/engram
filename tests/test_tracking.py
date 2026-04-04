"""Tests for experiment tracking: index and comparison."""

import json
from pathlib import Path

import pytest

from engram.models.scoring import EvalReport, FieldMetrics
from engram.tracking.comparison import FieldDelta, compare_experiments, diff_config_snapshots
from engram.tracking.index import append_to_index, read_index


def _setup_experiment(root: Path, experiment_id: str, impl: str, dataset: str, topic_output: str) -> None:
    """Create a minimal experiment with results and config snapshot."""
    exp_dir = root / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)

    results_data = {
        'experiment_id': experiment_id,
        'implementation': impl,
        'dataset': dataset,
        'timestamp': '2026-04-04T12:00:00Z',
        'total': 1,
        'succeeded': 1,
        'failed': 0,
        'results': [
            {
                'input_file': '001.txt',
                'output': {'topic': topic_output},
                'status': 'succeeded',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'cost_usd': 0.01,
                'latency_ms': 500,
                'error': '',
            }
        ],
    }
    (exp_dir / 'results.json').write_text(json.dumps(results_data))
    (exp_dir / 'config-snapshot.json').write_text(
        json.dumps({'models': ['claude-sonnet'], 'prompts': {'system.md': 'Classify.'}, 'runner_config': {}})
    )


def _setup_project_with_experiments(tmp_path: Path) -> tuple[str, str]:
    """Create a project with two experiments for comparison."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(
        'name: classify\n'
        'output:\n'
        '  fields:\n'
        '    topic:\n'
        '      type: enum\n'
        '      values: [A, B]\n'
        'scorers:\n'
        '  topic: exact_match\n'
    )

    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text('workflow: classify\nplatform: api\nrunner: anthropic\n')

    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    (ds_dir / 'labels.json').write_text(json.dumps({'001.txt': {'topic': 'A'}}))

    (tmp_path / 'experiments').mkdir(exist_ok=True)

    id_a = 'classify-api_test-ds_20260404_120000'
    id_b = 'classify-api_test-ds_20260404_130000'

    _setup_experiment(tmp_path, id_a, 'classify-api', 'test-ds', 'A')
    _setup_experiment(tmp_path, id_b, 'classify-api', 'test-ds', 'B')

    return id_a, id_b


# --- Index ---


def test_append_and_read_index(tmp_path: Path):
    (tmp_path / 'experiments').mkdir()

    # Create a minimal experiment for the index to read
    exp_id = 'test-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 1,
                'succeeded': 1,
                'failed': 0,
                'results': [],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    report = EvalReport(
        experiment_id=exp_id,
        field_metrics=[FieldMetrics(field_name='topic', accuracy=0.95, total=100, correct=95)],
        cost_total_usd=1.23,
        cost_avg_usd=0.0123,
    )

    append_to_index(tmp_path, report)

    entries = read_index(tmp_path)
    assert len(entries) == 1
    assert entries[0]['id'] == exp_id
    assert entries[0]['macro_accuracy'] == 0.95
    assert entries[0]['models'] == ['claude-sonnet']


def test_read_index_empty(tmp_path: Path):
    assert read_index(tmp_path) == []


# --- Field Delta ---


def test_field_delta():
    delta = FieldDelta(field_name='topic', accuracy_a=0.8, accuracy_b=0.9)
    assert delta.delta == pytest.approx(0.1)
    assert not delta.regressed

    delta_down = FieldDelta(field_name='sentiment', accuracy_a=0.9, accuracy_b=0.7)
    assert delta_down.delta == pytest.approx(-0.2)
    assert delta_down.regressed


# --- Config diff ---


def test_diff_config_snapshots(tmp_path: Path):
    id_a, id_b = 'exp-a', 'exp-b'

    exp_a = tmp_path / 'experiments' / id_a
    exp_b = tmp_path / 'experiments' / id_b
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)

    (exp_a / 'config-snapshot.json').write_text(
        json.dumps({'models': ['gpt-4'], 'prompts': {'system.md': 'v1'}, 'runner_config': {'max_tokens': '1000'}})
    )
    (exp_b / 'config-snapshot.json').write_text(
        json.dumps({'models': ['gpt-4.1'], 'prompts': {'system.md': 'v2'}, 'runner_config': {'max_tokens': '2000'}})
    )

    lines = diff_config_snapshots(tmp_path, id_a, id_b)
    assert any('Models' in line for line in lines)
    assert any('max_tokens' in line for line in lines)
    assert any('system.md' in line for line in lines)


# --- Comparison ---


def test_compare_experiments(tmp_path: Path):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    result = compare_experiments(tmp_path, id_a, id_b)

    assert result.experiment_a == id_a
    assert result.experiment_b == id_b
    assert 'topic' in result.field_deltas

    # A got topic right (A=A), B got it wrong (B!=A)
    assert result.field_deltas['topic'].accuracy_a == 1.0
    assert result.field_deltas['topic'].accuracy_b == 0.0
    assert result.regressions == ['topic']
