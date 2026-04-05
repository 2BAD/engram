"""Tests for scoring system: scorers, registry, engine, and metrics."""

import json
from pathlib import Path

import pytest

from engram.scoring.engine import score_experiment
from engram.scoring.metrics import compute_confusion_matrix, compute_cost_stats, compute_field_metrics
from engram.scoring.registry import resolve_scorer
from engram.scoring.scorers import exact_match, fuzzy_match, numeric_tolerance, set_match

# --- Built-in scorers ---


def test_exact_match():
    assert exact_match('Hello', 'hello')
    assert exact_match(' A ', 'a')
    assert not exact_match('A', 'B')


def test_fuzzy_match():
    scorer = fuzzy_match(0.8)
    assert scorer('hello world', 'hello worl')
    assert not scorer('hello', 'goodbye')


def test_set_match():
    assert set_match(['A', 'B'], ['B', 'A'])
    assert set_match('A, B', 'B, A')
    assert not set_match(['A', 'B'], ['A', 'C'])


def test_numeric_tolerance():
    scorer = numeric_tolerance(0.1)
    assert scorer(100, 105)  # within 10%
    assert scorer(95, 100)  # within 10%
    assert not scorer(80, 100)  # 20% off
    assert scorer(0, 0)
    assert not scorer('abc', 100)


# --- Scorer registry ---


def test_resolve_exact_match():
    scorer = resolve_scorer('exact_match')
    assert scorer('A', 'a')


def test_resolve_parameterized():
    scorer = resolve_scorer('numeric_tolerance(0.05)')
    assert scorer(100, 103)
    assert not scorer(100, 110)


def test_resolve_bare_fuzzy_match():
    scorer = resolve_scorer('fuzzy_match')
    assert scorer('hello world', 'hello worl')
    assert not scorer('hello', 'goodbye')


def test_resolve_bare_numeric_tolerance():
    scorer = resolve_scorer('numeric_tolerance')
    assert scorer(100, 105)
    assert not scorer(80, 100)


def test_resolve_unknown():
    with pytest.raises(ValueError, match='Unknown scorer'):
        resolve_scorer('nonexistent_scorer')


def test_resolve_custom_scorer(tmp_path: Path):
    scorer_code = 'def my_scorer(predicted, expected):\n    return predicted == expected\n'
    (tmp_path / 'scorers.py').write_text(scorer_code)

    scorer = resolve_scorer('scorers.my_scorer', workflow_dir=tmp_path)
    assert scorer('A', 'A')
    assert not scorer('A', 'B')


# --- Metrics ---


def test_compute_field_metrics():
    scores = [True, True, True, False, True]
    metrics = compute_field_metrics('topic', scores)
    assert metrics.field_name == 'topic'
    assert metrics.accuracy == 0.8
    assert metrics.total == 5
    assert metrics.correct == 4


def test_compute_field_metrics_empty():
    metrics = compute_field_metrics('topic', [])
    assert metrics.accuracy == 0.0
    assert metrics.total == 0


def test_compute_confusion_matrix():
    pairs = [('A', 'A'), ('A', 'B'), ('B', 'B'), ('B', 'A'), ('A', 'A')]
    cm = compute_confusion_matrix('topic', pairs)
    assert cm.labels == ['A', 'B']
    assert cm.matrix['A']['A'] == 2
    assert cm.matrix['A']['B'] == 1
    assert cm.matrix['B']['A'] == 1
    assert cm.matrix['B']['B'] == 1


def test_compute_cost_stats():
    costs = [0.01, 0.02, 0.03, 0.01, 0.05]
    total, avg, median, p95 = compute_cost_stats(costs)
    assert total == pytest.approx(0.12)
    assert avg == pytest.approx(0.024)
    assert median == pytest.approx(0.02)
    assert p95 == pytest.approx(0.05)


def test_compute_cost_stats_empty():
    assert compute_cost_stats([]) == (0.0, 0.0, 0.0, 0.0)


# --- Scoring engine integration ---


def _setup_scored_project(tmp_path: Path) -> str:
    """Create a project with workflow, implementation, dataset, and experiment results."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    # Workflow
    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(
        'name: classify\n'
        'output:\n'
        '  fields:\n'
        '    topic:\n'
        '      type: enum\n'
        '      values: [A, B, C]\n'
        'scorers:\n'
        '  topic: exact_match\n'
        'confusion_matrices:\n'
        '  - topic\n'
    )

    # Implementation
    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text('workflow: classify\nplatform: api\nrunner: anthropic\n')

    # Dataset with labels
    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    labels = {
        '001.txt': {'topic': 'A'},
        '002.txt': {'topic': 'B'},
        '003.txt': {'topic': 'A'},
    }
    (ds_dir / 'labels.json').write_text(json.dumps(labels))

    # Experiment results
    experiment_id = 'classify-api_test-ds_20260404_120000'
    exp_dir = tmp_path / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)

    results_data = {
        'experiment_id': experiment_id,
        'implementation': 'classify-api',
        'dataset': 'test-ds',
        'timestamp': '2026-04-04T12:00:00Z',
        'total': 3,
        'succeeded': 3,
        'failed': 0,
        'results': [
            {
                'input_file': '001.txt',
                'output': {'topic': 'A'},
                'status': 'succeeded',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'cost_usd': 0.01,
                'latency_ms': 500,
                'error': '',
            },
            {
                'input_file': '002.txt',
                'output': {'topic': 'B'},
                'status': 'succeeded',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'cost_usd': 0.01,
                'latency_ms': 600,
                'error': '',
            },
            {
                'input_file': '003.txt',
                'output': {'topic': 'C'},
                'status': 'succeeded',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'cost_usd': 0.02,
                'latency_ms': 700,
                'error': '',
            },
        ],
    }
    (exp_dir / 'results.json').write_text(json.dumps(results_data))

    return experiment_id


def test_score_experiment(tmp_path: Path):
    experiment_id = _setup_scored_project(tmp_path)
    report = score_experiment(tmp_path, experiment_id)

    assert report.experiment_id == experiment_id
    assert report.matched_examples == 3
    assert len(report.field_metrics) == 1

    topic_metrics = report.field_metrics[0]
    assert topic_metrics.field_name == 'topic'
    assert topic_metrics.total == 3
    assert topic_metrics.correct == 2  # 001=A(correct), 002=B(correct), 003=C(wrong, expected A)
    assert topic_metrics.accuracy == pytest.approx(2 / 3)

    assert len(report.confusion_matrices) == 1
    assert report.cost_total_usd == pytest.approx(0.04)
