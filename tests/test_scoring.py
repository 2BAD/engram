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


def test_compute_field_metrics_non_classification():
    """Without is_classification=True, P/R/F1 fall back to the accuracy value."""
    scores = [True, True, True, False, True]
    metrics = compute_field_metrics('topic', scores)
    assert metrics.field_name == 'topic'
    assert metrics.accuracy == 0.8
    assert metrics.precision == 0.8
    assert metrics.recall == 0.8
    assert metrics.f1 == 0.8
    assert metrics.total == 5
    assert metrics.correct == 4


def test_compute_field_metrics_empty():
    metrics = compute_field_metrics('topic', [])
    assert metrics.accuracy == 0.0
    assert metrics.total == 0


def test_compute_field_metrics_classification_perfect():
    """All-correct multi-class: every macro metric is 1.0."""
    scores = [True] * 5
    pairs = [('A', 'A'), ('A', 'A'), ('B', 'B'), ('B', 'B'), ('C', 'C')]
    metrics = compute_field_metrics('topic', scores, pairs=pairs, is_classification=True)
    assert metrics.accuracy == 1.0
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_compute_field_metrics_classification_all_wrong():
    """All-wrong 2-class: accuracy is 0, and because no class has any TP, macro P/R/F1 are 0."""
    scores = [False, False, False, False]
    pairs = [('A', 'B'), ('A', 'B'), ('B', 'A'), ('B', 'A')]
    metrics = compute_field_metrics('topic', scores, pairs=pairs, is_classification=True)
    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_compute_field_metrics_classification_symmetric_confusion():
    """3-class symmetric confusion. Hand-calculated expectations from comments."""
    # Class A: expected 2 (A,A and A,B), predicted 2 (A,A and B,A)
    #   TP=1 (A,A), FP=1 (B,A), FN=1 (A,B) → P=R=0.5, F1=0.5
    # Class B: expected 2 (B,B and B,A), predicted 2 (A,B and B,B)
    #   TP=1 (B,B), FP=1 (A,B), FN=1 (B,A) → P=R=0.5, F1=0.5
    # Class C: expected 1, predicted 1, both C → P=R=F1=1.0
    # Macro: (0.5 + 0.5 + 1.0) / 3 = 0.666...
    scores = [True, False, True, False, True]
    pairs = [('A', 'A'), ('A', 'B'), ('B', 'B'), ('B', 'A'), ('C', 'C')]
    metrics = compute_field_metrics('topic', scores, pairs=pairs, is_classification=True)
    assert metrics.accuracy == pytest.approx(3 / 5)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_compute_field_metrics_classification_hallucinated_class():
    """A class predicted but never expected (C) has P=R=F1=0 and drags down the macro average."""
    # Class A: expected 2 (A,A and A,C), predicted 1 (A,A)
    #   TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=2*1*0.5/1.5 = 0.6666...
    # Class B: expected 1, predicted 1 → P=R=F1=1.0
    # Class C: expected 0, predicted 1 → TP=0, FP=1, FN=0 → P=0 (0/1), R=0 (0/0 → 0), F1=0
    # Macro P: (1 + 1 + 0) / 3 = 0.6666...
    # Macro R: (0.5 + 1 + 0) / 3 = 0.5
    # Macro F1: (0.6666... + 1 + 0) / 3 = 0.5555...
    scores = [True, False, True]
    pairs = [('A', 'A'), ('A', 'C'), ('B', 'B')]
    metrics = compute_field_metrics('topic', scores, pairs=pairs, is_classification=True)
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx((2 / 3 + 1 + 0) / 3)


def test_compute_field_metrics_classification_imbalanced():
    """2-class imbalanced: majority class gets most predictions, minority class suffers."""
    # 5 pairs: [(A,A), (A,A), (A,A), (A,B), (B,A)]
    # Class A: expected 4, predicted 4 — TP=3, FP=1 (B,A), FN=1 (A,B) → P=R=0.75, F1=0.75
    # Class B: expected 1, predicted 1 — TP=0, FP=1 (A,B), FN=1 (B,A) → P=0, R=0, F1=0
    # Macro: P=R=F1=(0.75+0)/2=0.375
    # But accuracy is 3/5=0.6 — the gap between accuracy and macro F1 IS the point of macro.
    scores = [True, True, True, False, False]
    pairs = [('A', 'A'), ('A', 'A'), ('A', 'A'), ('A', 'B'), ('B', 'A')]
    metrics = compute_field_metrics('topic', scores, pairs=pairs, is_classification=True)
    assert metrics.accuracy == pytest.approx(0.6)
    assert metrics.precision == pytest.approx(0.375)
    assert metrics.recall == pytest.approx(0.375)
    assert metrics.f1 == pytest.approx(0.375)


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

    # Workflow uses enum + exact_match so per-class F1 is computed. Pairs are:
    #   [(A,A), (B,B), (A,C)]
    # Class A: TP=1, FP=0, FN=1 → P=1, R=0.5, F1=2*1*0.5/1.5 = 2/3
    # Class B: TP=1, FP=0, FN=0 → P=R=F1=1
    # Class C: expected 0, predicted 1 → TP=0, FP=1, FN=0 → P=R=F1=0
    # Macro P: (1+1+0)/3 = 2/3
    # Macro R: (0.5+1+0)/3 = 0.5
    # Macro F1: (2/3 + 1 + 0)/3 = 5/9 ≈ 0.556
    assert topic_metrics.precision == pytest.approx(2 / 3)
    assert topic_metrics.recall == pytest.approx(0.5)
    assert topic_metrics.f1 == pytest.approx(5 / 9)
    # enum + exact_match → real per-class metrics.
    assert topic_metrics.is_classification is True

    assert len(report.confusion_matrices) == 1
    assert report.cost_total_usd == pytest.approx(0.04)


def test_score_experiment_non_classification_field(tmp_path: Path):
    """A numeric field with numeric_tolerance scorer gets is_classification=False."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'measure'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(
        'name: measure\n'
        'output:\n'
        '  fields:\n'
        '    score:\n'
        '      type: number\n'
        'scorers:\n'
        '  score: numeric_tolerance(0.1)\n'
    )

    impl_dir = tmp_path / 'implementations' / 'measure-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text('workflow: measure\nplatform: api\nrunner: anthropic\n')

    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    (ds_dir / 'labels.json').write_text(json.dumps({'001.txt': {'score': 100}, '002.txt': {'score': 50}}))

    experiment_id = 'measure-api_test-ds_20260404_120000'
    exp_dir = tmp_path / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': experiment_id,
                'implementation': 'measure-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 2,
                'succeeded': 2,
                'failed': 0,
                'results': [
                    {
                        'input_file': '001.txt',
                        'output': {'score': 105},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                        'cost_usd': 0.0,
                        'latency_ms': 100,
                        'error': '',
                    },
                    {
                        'input_file': '002.txt',
                        'output': {'score': 60},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                        'cost_usd': 0.0,
                        'latency_ms': 100,
                        'error': '',
                    },
                ],
            }
        )
    )

    report = score_experiment(tmp_path, experiment_id)
    score_metrics = report.field_metrics[0]
    # 105 is within 10% of 100 (correct), 60 is not within 10% of 50 (off by 20%) → 1/2.
    assert score_metrics.accuracy == pytest.approx(0.5)
    # Non-classification: P/R/F1 fall back to accuracy, and the flag is False.
    assert score_metrics.is_classification is False
    assert score_metrics.precision == pytest.approx(0.5)
    assert score_metrics.recall == pytest.approx(0.5)
    assert score_metrics.f1 == pytest.approx(0.5)
