"""Scoring engine: score experiment results against labels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.config.loader import load_implementation, load_workflow
from engram.datasets.loader import load_dataset_labels
from engram.eval.results import load_results
from engram.models.scoring import EvalReport, FieldMetrics
from engram.scoring.metrics import compute_confusion_matrix, compute_cost_stats, compute_field_metrics
from engram.scoring.registry import resolve_scorer

if TYPE_CHECKING:
    from engram.models.run import RunResult


def score_experiment(root: Path, experiment_id: str) -> EvalReport:
    """Score an experiment's results against dataset labels."""
    exp_dir = root / 'experiments' / experiment_id
    metadata, results = load_results(exp_dir)

    impl_config = load_implementation(root, metadata['implementation'])
    wf = load_workflow(root, impl_config.workflow)
    workflow_dir = root / 'workflows' / impl_config.workflow

    labels = load_dataset_labels(root, metadata['dataset'])
    resolved_scorers = {name: resolve_scorer(scorer, workflow_dir) for name, scorer in wf.scorers.items()}

    field_scores, field_predictions, matched_examples = _collect_scores(results, labels, resolved_scorers)

    # A field gets per-class precision/recall/F1 only when the scorer is exact_match
    # against an enum output — that's the regime where "class" is well-defined. Fuzzy,
    # numeric, and custom scorers fall back to accuracy-based values.
    classification_fields = {
        name
        for name, scorer_ref in wf.scorers.items()
        if wf.output_fields.get(name) is not None
        and wf.output_fields[name].type == 'enum'
        and scorer_ref == 'exact_match'
    }

    all_field_metrics = [
        compute_field_metrics(
            name,
            scores,
            pairs=field_predictions.get(name, []),
            is_classification=name in classification_fields,
        )
        if scores
        else FieldMetrics(field_name=name)
        for name, scores in field_scores.items()
    ]

    confusion_matrices = [
        compute_confusion_matrix(name, pairs)
        for name in wf.confusion_matrices
        if (pairs := field_predictions.get(name, []))
    ]

    costs = [r.cost_usd for r in results if r.status == 'succeeded' and r.cost_usd > 0]
    cost_total, cost_avg, cost_median, cost_p95 = compute_cost_stats(costs)

    return EvalReport(
        experiment_id=experiment_id,
        matched_examples=matched_examples,
        field_metrics=all_field_metrics,
        confusion_matrices=confusion_matrices,
        cost_total_usd=cost_total,
        cost_avg_usd=cost_avg,
        cost_median_usd=cost_median,
        cost_p95_usd=cost_p95,
    )


def _collect_scores(
    results: list[RunResult],
    labels: dict[str, dict[str, Any]],
    scorers: dict[str, Any],
) -> tuple[dict[str, list[bool]], dict[str, list[tuple[str, str]]], int]:
    """Score each result and collect (expected, predicted) pairs for every scored field."""
    field_scores: dict[str, list[bool]] = {f: [] for f in scorers}
    field_predictions: dict[str, list[tuple[str, str]]] = {f: [] for f in scorers}
    matched_examples = 0

    for result in results:
        if result.status != 'succeeded':
            continue
        example_labels = labels.get(result.input_file, {})
        if not example_labels:
            continue
        matched_examples += 1
        _score_single_result(result, example_labels, scorers, field_scores, field_predictions)

    return field_scores, field_predictions, matched_examples


def _score_single_result(
    result: RunResult,
    example_labels: dict[str, Any],
    scorers: dict[str, Any],
    field_scores: dict[str, list[bool]],
    field_predictions: dict[str, list[tuple[str, str]]],
) -> None:
    """Apply scorers to a single result against its labels."""
    for field_name, scorer_fn in scorers.items():
        if field_name not in example_labels:
            continue
        predicted = result.output.get(field_name)
        expected = example_labels[field_name]
        if predicted is None:
            continue
        field_scores[field_name].append(scorer_fn(predicted, expected))
        field_predictions[field_name].append((str(expected), str(predicted)))
