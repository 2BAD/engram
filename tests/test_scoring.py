"""Tests for scoring system: scorers, registry, engine, and metrics."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engram.cli import app
from engram.scoring.engine import load_saved_report, score_experiment
from engram.scoring.metrics import (
    compute_accuracy_stdev,
    compute_agreement_metrics,
    compute_confusion_matrix,
    compute_cost_stats,
    compute_field_metrics,
)
from engram.scoring.registry import resolve_scorer
from engram.scoring.scorers import (
    contains,
    contains_all,
    contains_any,
    exact_match,
    fuzzy_match,
    json_match,
    numeric_tolerance,
    regex,
    set_match,
)

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


def test_contains():
    assert contains('The topic is Finance and Banking', 'finance')
    assert contains('  HELLO WORLD  ', 'hello')
    assert not contains('The topic is Finance', 'healthcare')
    assert contains(123, '12')


def test_contains_all_with_list():
    assert contains_all('The topic is Finance and the sentiment is Positive', ['finance', 'positive'])
    assert not contains_all('The topic is Finance', ['finance', 'healthcare'])


def test_contains_all_with_csv_string():
    assert contains_all('Finance and Positive outcome', 'finance, positive')
    assert not contains_all('Finance only', 'finance, positive')


def test_contains_all_empty_expected():
    assert contains_all('anything', [])
    assert contains_all('anything', '')


def test_contains_any_with_list():
    assert contains_any('The topic is Finance', ['finance', 'healthcare'])
    assert not contains_any('The topic is Sports', ['finance', 'healthcare'])


def test_contains_any_with_csv_string():
    assert contains_any('Positive sentiment', 'negative, positive')
    assert not contains_any('Neutral sentiment', 'negative, positive')


def test_contains_any_empty_expected():
    assert not contains_any('anything', [])
    assert not contains_any('anything', '')


def test_numeric_tolerance():
    scorer = numeric_tolerance(0.1)
    assert scorer(100, 105)  # within 10%
    assert scorer(95, 100)  # within 10%
    assert not scorer(80, 100)  # 20% off
    assert scorer(0, 0)
    assert not scorer('abc', 100)


def test_json_match_dict_key_order():
    scorer = json_match()
    assert scorer({'name': 'Alice', 'age': 30}, {'age': 30, 'name': 'Alice'})
    assert not scorer({'name': 'Alice'}, {'name': 'Bob'})


def test_json_match_from_strings():
    scorer = json_match()
    assert scorer('{"a": 1, "b": 2}', '{"b": 2, "a": 1}')
    assert not scorer('{"a": 1}', '{"a": 2}')


def test_json_match_mixed_string_and_dict():
    scorer = json_match()
    assert scorer('{"x": 10}', {'x': 10})
    assert scorer({'x': 10}, '{"x": 10}')


def test_json_match_nested():
    scorer = json_match()
    assert scorer(
        {'outer': {'b': 2, 'a': 1}, 'list': [1, 2]},
        {'outer': {'a': 1, 'b': 2}, 'list': [1, 2]},
    )
    # List order matters
    assert not scorer({'list': [2, 1]}, {'list': [1, 2]})


def test_json_match_ignore_extra():
    scorer = json_match(ignore_extra=True)
    assert scorer({'name': 'Alice', 'age': 30, 'extra': 'field'}, {'name': 'Alice', 'age': 30})
    # Missing expected key fails
    assert not scorer({'name': 'Alice'}, {'name': 'Alice', 'age': 30})


def test_json_match_ignore_extra_nested():
    scorer = json_match(ignore_extra=True)
    assert scorer(
        {'user': {'name': 'Alice', 'id': 99}, 'meta': 'ignored'},
        {'user': {'name': 'Alice'}},
    )


def test_json_match_non_json_strings():
    scorer = json_match()
    # Plain strings that aren't valid JSON compare as strings
    assert scorer('hello', 'hello')
    assert not scorer('hello', 'world')


def test_json_match_primitives():
    scorer = json_match()
    assert scorer(42, 42)
    assert scorer('true', True)
    assert scorer('null', None)
    assert not scorer(42, 43)


def test_regex_basic():
    scorer = regex()
    assert scorer('order #12345 confirmed', r'\d{5}')
    assert not scorer('no numbers here', r'\d{5}')


def test_regex_case_insensitive():
    scorer = regex(flags='i')
    assert scorer('Hello World', r'hello')
    scorer_strict = regex()
    assert not scorer_strict('Hello World', r'hello')


def test_regex_anchored():
    scorer = regex()
    assert scorer('abc123', r'^abc\d+$')
    assert not scorer('xabc123', r'^abc\d+$')


def test_regex_multiline():
    scorer = regex(flags='m')
    assert scorer('line1\nstart here', r'^start')


def test_regex_dotall():
    scorer = regex(flags='s')
    assert scorer('line1\nline2', r'line1.line2')
    scorer_no_dotall = regex()
    assert not scorer_no_dotall('line1\nline2', r'line1.line2')


def test_regex_combined_flags():
    scorer = regex(flags='is')
    assert scorer('Line1\nline2', r'LINE1.line2')


def test_regex_coerces_to_string():
    scorer = regex()
    assert scorer(12345, r'^\d+$')


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


def test_resolve_contains():
    scorer = resolve_scorer('contains')
    assert scorer('The topic is Finance', 'finance')
    assert not scorer('Sports news', 'finance')


def test_resolve_contains_all():
    scorer = resolve_scorer('contains_all')
    assert scorer('Finance and Positive', ['finance', 'positive'])


def test_resolve_contains_any():
    scorer = resolve_scorer('contains_any')
    assert scorer('Finance report', ['finance', 'healthcare'])
    assert not scorer('Sports report', ['finance', 'healthcare'])


def test_resolve_json_match():
    scorer = resolve_scorer('json_match')
    assert scorer({'a': 1}, {'a': 1})


def test_resolve_json_match_parameterized():
    scorer = resolve_scorer('json_match(true)')
    assert scorer({'a': 1, 'b': 2}, {'a': 1})
    assert not scorer({'a': 1}, {'a': 1, 'b': 2})


def test_resolve_regex():
    scorer = resolve_scorer('regex')
    assert scorer('abc123', r'\d+')


def test_resolve_regex_parameterized():
    scorer = resolve_scorer('regex(i)')
    assert scorer('HELLO', r'hello')


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


# --- Repeat-aware metrics ---


def test_agreement_metrics_perfect_agreement():
    """All 3 repeats agree on every input → mean=1.0, majority=1.0, kappa=1.0."""
    predictions = {
        '001.txt': ['A', 'A', 'A'],
        '002.txt': ['B', 'B', 'B'],
        '003.txt': ['A', 'A', 'A'],
    }
    mean, majority, kappa = compute_agreement_metrics(predictions, repeats=3)
    assert mean == pytest.approx(1.0)
    assert majority == pytest.approx(1.0)
    assert kappa == pytest.approx(1.0)


def test_agreement_metrics_split_decision():
    """3 repeats with one input fully split (A,B,A → mode=A count=2; majority since 2>1.5)."""
    predictions = {
        '001.txt': ['A', 'A', 'A'],  # mode count 3, agreement 1.0
        '002.txt': ['A', 'B', 'A'],  # mode count 2, agreement 2/3
        '003.txt': ['A', 'B', 'C'],  # mode count 1, agreement 1/3, no majority
    }
    mean, majority, kappa = compute_agreement_metrics(predictions, repeats=3)
    assert mean == pytest.approx((1.0 + 2 / 3 + 1 / 3) / 3)
    # 2 of 3 inputs had a strict majority (>N/2): the all-A and the A,B,A.
    assert majority == pytest.approx(2 / 3)
    # Kappa is bounded; we sanity-check sign and that it's between -1 and 1.
    assert kappa is not None
    assert -1.0 <= kappa <= 1.0


def test_agreement_metrics_fleiss_known_value():
    """Hand-verified Fleiss kappa for a 2-item, 3-rater, 2-category case."""
    # Item 1: 3 raters all picked A → row [3, 0]
    # Item 2: 2 raters picked A, 1 picked B → row [2, 1]
    # n=3, P_1 = (9 - 3) / (3*2) = 1.0
    # P_2 = (4 + 1 - 3) / (3*2) = 2/6 = 0.333
    # P_bar = (1.0 + 0.333) / 2 = 0.667
    # p_A = (3+2)/(2*3) = 5/6, p_B = 1/6
    # P_e = (5/6)^2 + (1/6)^2 = 25/36 + 1/36 = 26/36 = 0.722
    # kappa = (0.667 - 0.722) / (1 - 0.722) = -0.055 / 0.278 ≈ -0.2
    predictions = {
        'item1': ['A', 'A', 'A'],
        'item2': ['A', 'A', 'B'],
    }
    _, _, kappa = compute_agreement_metrics(predictions, repeats=3)
    assert kappa == pytest.approx(-0.2, abs=0.01)


def test_agreement_metrics_collapsed_to_one_label_returns_kappa_one():
    """Edge case: every item is unanimous on the same label. Kappa is mathematically undefined; report 1.0."""
    predictions = {
        '001.txt': ['A', 'A', 'A'],
        '002.txt': ['A', 'A', 'A'],
    }
    _, _, kappa = compute_agreement_metrics(predictions, repeats=3)
    assert kappa == pytest.approx(1.0)


def test_agreement_metrics_two_repeats_omits_majority():
    """N=2 has no strict majority concept; majority_rate is None but mean and kappa still computed."""
    predictions = {'001.txt': ['A', 'A'], '002.txt': ['A', 'B']}
    mean, majority, kappa = compute_agreement_metrics(predictions, repeats=2)
    assert mean is not None
    assert majority is None
    assert kappa is not None


def test_agreement_metrics_single_repeat_returns_all_none():
    """repeats=1 short-circuits — there's nothing to compare."""
    assert compute_agreement_metrics({'001.txt': ['A']}, repeats=1) == (None, None, None)


def test_agreement_metrics_fleiss_skips_partial_items():
    """An item with fewer than `repeats` predictions is excluded from kappa but still contributes to mean."""
    predictions = {
        '001.txt': ['A', 'A', 'A'],  # full
        '002.txt': ['B', 'B', 'B'],  # full
        '003.txt': ['A', 'A'],  # one repeat failed; excluded from kappa only
    }
    mean, _, kappa = compute_agreement_metrics(predictions, repeats=3)
    # Mean includes all three: (1 + 1 + 1) / 3 = 1.0
    assert mean == pytest.approx(1.0)
    # Kappa computed over the 2 full items, both unanimous on different labels.
    # Each item: P_i = 1.0 for the row of all 3 in one category.
    # P_bar = 1.0
    # p_A = 3/6 = 0.5, p_B = 3/6 = 0.5; P_e = 0.5
    # kappa = (1.0 - 0.5) / (1 - 0.5) = 1.0
    assert kappa == pytest.approx(1.0)


def test_accuracy_stdev_across_repeats():
    """Stdev across per-repeat field accuracies."""
    per_repeat = {
        0: [True, True, False, False],  # 0.5
        1: [True, True, True, False],  # 0.75
        2: [True, True, True, True],  # 1.0
    }
    stdev = compute_accuracy_stdev(per_repeat)
    # statistics.stdev([0.5, 0.75, 1.0]) = 0.25
    assert stdev == pytest.approx(0.25)


def test_accuracy_stdev_returns_none_when_only_one_repeat():
    assert compute_accuracy_stdev({0: [True, False]}) is None


def test_accuracy_stdev_skips_empty_repeats():
    """A repeat that produced no scored examples doesn't enter the stdev computation."""
    per_repeat = {
        0: [True, True],  # 1.0
        1: [],  # skipped
        2: [False, False],  # 0.0
    }
    stdev = compute_accuracy_stdev(per_repeat)
    # statistics.stdev([1.0, 0.0])
    assert stdev == pytest.approx(0.7071, abs=0.001)


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
    # No cache tokens recorded → cache_hit_rate stays None so the display can hide the row.
    assert report.cache_hit_rate is None

    # Labels fingerprint is populated from the dataset used at score time.
    assert len(report.labels_hash) == 64  # SHA256 hex digest
    assert report.labels_count == 3
    assert report.labels_scored_at


def test_score_experiment_cost_breakdown_sums_to_total(tmp_path: Path):
    """When results carry per-bucket cost fields, EvalReport aggregates them per bucket and the sum equals total."""
    experiment_id = _setup_scored_project(tmp_path)

    # Each run carries a small breakdown; total is just the sum of the four components.
    results_path = tmp_path / 'experiments' / experiment_id / 'results.json'
    data = json.loads(results_path.read_text())
    for r in data['results']:
        r['cost_input_usd'] = 0.002
        r['cost_cache_read_usd'] = 0.001
        r['cost_cache_creation_usd'] = 0.0005
        r['cost_output_usd'] = 0.003
        r['cost_usd'] = 0.0065  # = sum of the four
    results_path.write_text(json.dumps(data))

    report = score_experiment(tmp_path, experiment_id)
    assert report.cost_input_usd == pytest.approx(0.006)
    assert report.cost_cache_read_usd == pytest.approx(0.003)
    assert report.cost_cache_creation_usd == pytest.approx(0.0015)
    assert report.cost_output_usd == pytest.approx(0.009)
    assert report.cost_total_usd == pytest.approx(
        report.cost_input_usd + report.cost_cache_read_usd + report.cost_cache_creation_usd + report.cost_output_usd
    )


def test_score_experiment_cache_hit_rate(tmp_path: Path):
    """When results record cache_read_tokens, the report exposes the aggregate hit rate."""
    experiment_id = _setup_scored_project(tmp_path)

    # Rewrite results.json with non-zero cache_read_tokens on two of the three runs.
    results_path = tmp_path / 'experiments' / experiment_id / 'results.json'
    data = json.loads(results_path.read_text())
    data['results'][0]['usage'] = {
        'prompt_tokens': 1000,
        'completion_tokens': 50,
        'total_tokens': 1050,
        'cache_read_tokens': 800,
        'cache_creation_tokens': 0,
    }
    data['results'][1]['usage'] = {
        'prompt_tokens': 1000,
        'completion_tokens': 50,
        'total_tokens': 1050,
        'cache_read_tokens': 600,
        'cache_creation_tokens': 0,
    }
    data['results'][2]['usage'] = {
        'prompt_tokens': 1000,
        'completion_tokens': 50,
        'total_tokens': 1050,
        'cache_read_tokens': 0,
        'cache_creation_tokens': 200,
    }
    results_path.write_text(json.dumps(data))

    report = score_experiment(tmp_path, experiment_id)
    # (800 + 600 + 0) / (1000 + 1000 + 1000) = 1400 / 3000
    assert report.cache_hit_rate == pytest.approx(1400 / 3000)


def test_score_experiment_labels_hash_roundtrip(tmp_path: Path):
    """Saving a report and reloading it preserves the labels fingerprint fields."""
    experiment_id = _setup_scored_project(tmp_path)
    report = score_experiment(tmp_path, experiment_id)

    eval_path = tmp_path / 'experiments' / experiment_id / 'eval.json'
    eval_path.write_text(json.dumps(asdict(report), indent=2))

    loaded = load_saved_report(tmp_path, experiment_id)
    assert loaded is not None
    assert loaded.labels_hash == report.labels_hash
    assert loaded.labels_count == report.labels_count
    assert loaded.labels_scored_at == report.labels_scored_at


def test_score_experiment_labels_hash_changes_with_labels(tmp_path: Path):
    """Editing labels.json produces a different labels_hash on the next score pass."""
    experiment_id = _setup_scored_project(tmp_path)
    before = score_experiment(tmp_path, experiment_id).labels_hash

    labels_path = tmp_path / 'datasets' / 'test-ds' / 'labels.json'
    labels = json.loads(labels_path.read_text())
    labels['003.txt']['topic'] = 'C'  # reclassify one entry
    labels_path.write_text(json.dumps(labels))

    after = score_experiment(tmp_path, experiment_id).labels_hash
    assert before != after


def test_score_experiment_non_classification_field(tmp_path: Path):
    """A numeric field with numeric_tolerance scorer gets is_classification=False."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'measure'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(
        'name: measure\noutput:\n  fields:\n    score:\n      type: number\nscorers:\n  score: numeric_tolerance(0.1)\n'
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


def test_score_experiment_with_repeats_populates_agreement_metrics(tmp_path: Path):
    """End-to-end: a 3-repeat experiment scores cleanly with agreement metrics on the FieldMetrics."""
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
    (ds_dir / 'labels.json').write_text(json.dumps({'001.txt': {'topic': 'A'}, '002.txt': {'topic': 'B'}}))

    experiment_id = 'classify-api_test-ds_repeats'
    exp_dir = tmp_path / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)

    def _result(input_file: str, repeat_index: int, topic: str) -> dict:
        return {
            'input_file': input_file,
            'output': {'topic': topic},
            'status': 'succeeded',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            'cost_usd': 0.0,
            'latency_ms': 100,
            'error': '',
            'repeat_index': repeat_index,
        }

    # Input 001 (label A): all 3 repeats agree on A.
    # Input 002 (label B): repeats split A, B, B → modal answer is B (correct), agreement 2/3.
    results_data = {
        'experiment_id': experiment_id,
        'implementation': 'classify-api',
        'dataset': 'test-ds',
        'timestamp': '2026-04-04T12:00:00Z',
        'total': 6,
        'succeeded': 6,
        'failed': 0,
        'results': [
            _result('001.txt', 0, 'A'),
            _result('001.txt', 1, 'A'),
            _result('001.txt', 2, 'A'),
            _result('002.txt', 0, 'A'),
            _result('002.txt', 1, 'B'),
            _result('002.txt', 2, 'B'),
        ],
    }
    (exp_dir / 'results.json').write_text(json.dumps(results_data))

    report = score_experiment(tmp_path, experiment_id)
    topic = report.field_metrics[0]

    # Pooled accuracy: 5 of 6 scored predictions correct = 5/6.
    assert topic.accuracy == pytest.approx(5 / 6)

    # Repeat-aware metrics are all populated.
    assert topic.accuracy_stdev is not None
    assert topic.mean_agreement_rate == pytest.approx((1.0 + 2 / 3) / 2)
    assert topic.majority_rate == pytest.approx(1.0)  # both inputs had a strict majority
    assert topic.fleiss_kappa is not None


def test_score_experiment_single_repeat_omits_agreement_metrics(tmp_path: Path):
    """A single-repeat experiment leaves all four new fields as None — backward compat (D12)."""
    experiment_id = _setup_scored_project(tmp_path)
    report = score_experiment(tmp_path, experiment_id)
    topic = report.field_metrics[0]
    assert topic.mean_agreement_rate is None
    assert topic.majority_rate is None
    assert topic.fleiss_kappa is None
    assert topic.accuracy_stdev is None


def test_score_command_json_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With --json, score emits an EvalReport as structured JSON."""
    experiment_id = _setup_scored_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ['--json', 'score', experiment_id])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert payload['experiment_id'] == experiment_id
    assert payload['matched_examples'] == 3
    # Field metrics serialized as a list of dicts with the full per-metric shape.
    topic = next(fm for fm in payload['field_metrics'] if fm['field_name'] == 'topic')
    assert topic['accuracy'] == pytest.approx(2 / 3)
    assert topic['f1'] == pytest.approx(5 / 9)
    assert topic['is_classification'] is True
    # Confusion matrices and cost stats round-trip too.
    assert len(payload['confusion_matrices']) == 1
    assert payload['cost_total_usd'] == pytest.approx(0.04)


@pytest.mark.usefixtures('rich_mode')
def test_score_save_notes_drift_when_labels_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Editing labels between two --save passes surfaces a drift notice and rewrites eval.json."""
    experiment_id = _setup_scored_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    CliRunner().invoke(app, ['score', experiment_id, '--save'])

    labels_path = tmp_path / 'datasets' / 'test-ds' / 'labels.json'
    labels = json.loads(labels_path.read_text())
    labels['003.txt']['topic'] = 'C'
    labels_path.write_text(json.dumps(labels))

    second = CliRunner().invoke(app, ['score', experiment_id, '--save'])
    assert second.exit_code == 0
    assert 'Labels changed since last score' in second.output
    assert 'Saved eval report' in second.output
