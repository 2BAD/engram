"""Scoring engine: score experiment results against labels."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.config.loader import load_implementation, load_workflow
from engram.datasets.loader import compute_labels_hash, load_dataset_inputs, load_dataset_labels
from engram.eval.results import load_results
from engram.models.scoring import ConfusionMatrix, EvalReport, FieldMetrics
from engram.scoring.metrics import (
    compute_accuracy_stdev,
    compute_agreement_metrics,
    compute_confusion_matrix,
    compute_cost_stats,
    compute_field_metrics,
)
from engram.scoring.registry import resolve_scorer, scorer_accepts_input_data

if TYPE_CHECKING:
    from engram.models.input import InputData
    from engram.models.run import RunResult


def load_saved_report(root: Path, experiment_id: str) -> EvalReport | None:
    """Load a previously saved eval report, or None if not available."""
    eval_path = root / 'experiments' / experiment_id / 'eval.json'
    if not eval_path.exists():
        return None
    import json  # noqa: PLC0415

    raw = json.loads(eval_path.read_text())
    return EvalReport(
        experiment_id=raw['experiment_id'],
        matched_examples=raw.get('matched_examples', 0),
        successful_calls=raw.get('successful_calls', 0),
        field_metrics=[FieldMetrics(**fm) for fm in raw.get('field_metrics', [])],
        confusion_matrices=[ConfusionMatrix(**cm) for cm in raw.get('confusion_matrices', [])],
        cost_total_usd=raw.get('cost_total_usd', 0.0),
        cost_avg_usd=raw.get('cost_avg_usd', 0.0),
        cost_median_usd=raw.get('cost_median_usd', 0.0),
        cost_p95_usd=raw.get('cost_p95_usd', 0.0),
        cost_input_usd=raw.get('cost_input_usd', 0.0),
        cost_cache_read_usd=raw.get('cost_cache_read_usd', 0.0),
        cost_cache_creation_usd=raw.get('cost_cache_creation_usd', 0.0),
        cost_output_usd=raw.get('cost_output_usd', 0.0),
        cost_without_cache_total_usd=raw.get('cost_without_cache_total_usd', 0.0),
        tokens_prompt=raw.get('tokens_prompt', 0),
        tokens_cache_read=raw.get('tokens_cache_read', 0),
        tokens_cache_creation=raw.get('tokens_cache_creation', 0),
        tokens_completion=raw.get('tokens_completion', 0),
        cache_hit_rate=raw.get('cache_hit_rate'),
        labels_hash=raw.get('labels_hash', ''),
        labels_count=raw.get('labels_count', 0),
        labels_scored_at=raw.get('labels_scored_at', ''),
    )


def score_experiment(root: Path, experiment_id: str) -> EvalReport:
    """Score an experiment's results against dataset labels."""
    exp_dir = root / 'experiments' / experiment_id
    metadata, results = load_results(exp_dir)

    impl_config = load_implementation(root, metadata['implementation'])
    wf = load_workflow(root, impl_config.workflow)
    workflow_dir = root / 'workflows' / impl_config.workflow

    labels = load_dataset_labels(root, metadata['dataset'])
    resolved_scorers = {name: resolve_scorer(scorer, workflow_dir) for name, scorer in wf.scorers.items()}

    # Load dataset inputs only when at least one scorer asks for them (e.g. an LLM judge that
    # grades a summary against its source). Avoids reading binary files into memory otherwise.
    inputs_by_filename: dict[str, InputData] = {}
    if any(scorer_accepts_input_data(fn) for fn in resolved_scorers.values()):
        inputs_by_filename = {inp.filename: inp for inp in load_dataset_inputs(root, metadata['dataset'])}

    repeats = max((r.repeat_index for r in results), default=0) + 1
    field_scores, field_predictions, per_repeat_scores, predictions_by_input, matched_examples = _collect_scores(
        results, labels, resolved_scorers, inputs_by_filename
    )

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
        _build_field_metrics(
            name,
            scores,
            pairs=field_predictions.get(name, []),
            is_classification=name in classification_fields,
            per_repeat=per_repeat_scores.get(name, {}),
            per_input_predictions=predictions_by_input.get(name, {}),
            repeats=repeats,
        )
        for name, scores in field_scores.items()
    ]

    confusion_matrices = [
        compute_confusion_matrix(name, pairs)
        for name in wf.confusion_matrices
        if (pairs := field_predictions.get(name, []))
    ]

    costs = [r.cost_usd for r in results if r.status == 'succeeded' and r.cost_usd > 0]
    cost_total, cost_avg, cost_median, cost_p95 = compute_cost_stats(costs)

    successful = [r for r in results if r.status == 'succeeded']
    cache_hit_rate = _compute_cache_hit_rate(successful)
    cost_input = sum(r.cost_input_usd for r in successful)
    cost_cache_read = sum(r.cost_cache_read_usd for r in successful)
    cost_cache_creation = sum(r.cost_cache_creation_usd for r in successful)
    cost_output = sum(r.cost_output_usd for r in successful)
    cost_without_cache = sum(r.cost_without_cache_usd for r in successful)
    tokens_prompt = sum(r.usage.prompt_tokens for r in successful)
    tokens_cache_read = sum(r.usage.cache_read_tokens for r in successful)
    tokens_cache_creation = sum(r.usage.cache_creation_tokens for r in successful)
    tokens_completion = sum(r.usage.completion_tokens for r in successful)

    return EvalReport(
        experiment_id=experiment_id,
        matched_examples=matched_examples,
        successful_calls=len(successful),
        field_metrics=all_field_metrics,
        confusion_matrices=confusion_matrices,
        cost_total_usd=cost_total,
        cost_avg_usd=cost_avg,
        cost_median_usd=cost_median,
        cost_p95_usd=cost_p95,
        cost_input_usd=cost_input,
        cost_cache_read_usd=cost_cache_read,
        cost_cache_creation_usd=cost_cache_creation,
        cost_output_usd=cost_output,
        cost_without_cache_total_usd=cost_without_cache,
        tokens_prompt=tokens_prompt,
        tokens_cache_read=tokens_cache_read,
        tokens_cache_creation=tokens_cache_creation,
        tokens_completion=tokens_completion,
        cache_hit_rate=cache_hit_rate,
        labels_hash=compute_labels_hash(labels),
        labels_count=len(labels),
        labels_scored_at=datetime.now(UTC).isoformat(),
    )


def _compute_cache_hit_rate(results: list[RunResult]) -> float | None:
    """Aggregate cache-read tokens over total prompt tokens. None when neither is recorded."""
    total_prompt = sum(r.usage.prompt_tokens for r in results)
    total_cache_read = sum(r.usage.cache_read_tokens for r in results)
    total_cache_creation = sum(r.usage.cache_creation_tokens for r in results)
    if total_cache_read == 0 and total_cache_creation == 0:
        return None
    if total_prompt == 0:
        return None
    return total_cache_read / total_prompt


def _collect_scores(
    results: list[RunResult],
    labels: dict[str, dict[str, Any]],
    scorers: dict[str, Any],
    inputs_by_filename: dict[str, InputData],
) -> tuple[
    dict[str, list[bool]],
    dict[str, list[tuple[str, str]]],
    dict[str, dict[int, list[bool]]],
    dict[str, dict[str, list[str]]],
    int,
]:
    """
    Score each result and collect everything the metric pass needs.

    Returns a 5-tuple:
    - field_scores: pooled list of bool scores per field across all repeats (for accuracy/P/R/F1)
    - field_predictions: list of (expected, predicted) string pairs per field (for confusion matrix
      and classification metrics)
    - per_repeat_scores: per field, per repeat_index, the list of bool scores (for accuracy_stdev)
    - predictions_by_input: per field, per input_file, the list of stringified predictions (for
      agreement metrics)
    - matched_examples: count of distinct (input_file, repeat_index) pairs that had at least one
      labeled field
    """
    field_scores: dict[str, list[bool]] = {f: [] for f in scorers}
    field_predictions: dict[str, list[tuple[str, str]]] = {f: [] for f in scorers}
    per_repeat_scores: dict[str, dict[int, list[bool]]] = {f: {} for f in scorers}
    predictions_by_input: dict[str, dict[str, list[str]]] = {f: {} for f in scorers}
    matched_examples = 0

    for result in results:
        if result.status != 'succeeded':
            continue
        example_labels = labels.get(result.input_file, {})
        if not example_labels:
            continue
        matched_examples += 1
        _score_single_result(
            result,
            example_labels,
            scorers,
            field_scores,
            field_predictions,
            per_repeat_scores,
            predictions_by_input,
            inputs_by_filename.get(result.input_file),
        )

    return field_scores, field_predictions, per_repeat_scores, predictions_by_input, matched_examples


def _score_single_result(  # noqa: PLR0913 — internal helper that fans one result into four parallel accumulators
    result: RunResult,
    example_labels: dict[str, Any],
    scorers: dict[str, Any],
    field_scores: dict[str, list[bool]],
    field_predictions: dict[str, list[tuple[str, str]]],
    per_repeat_scores: dict[str, dict[int, list[bool]]],
    predictions_by_input: dict[str, dict[str, list[str]]],
    input_data: InputData | None,
) -> None:
    """Apply scorers to a single result against its labels."""
    for field_name, scorer_fn in scorers.items():
        if field_name not in example_labels:
            continue
        predicted = result.output.get(field_name)
        expected = example_labels[field_name]
        if predicted is None:
            continue
        score = (
            scorer_fn(predicted, expected, input_data=input_data)
            if scorer_accepts_input_data(scorer_fn)
            else scorer_fn(predicted, expected)
        )
        field_scores[field_name].append(score)
        field_predictions[field_name].append((str(expected), str(predicted)))
        per_repeat_scores[field_name].setdefault(result.repeat_index, []).append(score)
        predictions_by_input[field_name].setdefault(result.input_file, []).append(str(predicted))


def _build_field_metrics(  # noqa: PLR0913 — composes seven facts about one field into a single result struct
    name: str,
    scores: list[bool],
    pairs: list[tuple[str, str]],
    is_classification: bool,
    per_repeat: dict[int, list[bool]],
    per_input_predictions: dict[str, list[str]],
    repeats: int,
) -> FieldMetrics:
    """Compose accuracy/P/R/F1 with the repeat-aware metrics, leaving the latter None when repeats <= 1."""
    if not scores:
        return FieldMetrics(field_name=name)

    metrics = compute_field_metrics(name, scores, pairs=pairs, is_classification=is_classification)

    if repeats > 1:
        mean_agreement, majority, kappa = compute_agreement_metrics(per_input_predictions, repeats)
        metrics.mean_agreement_rate = mean_agreement
        metrics.majority_rate = majority
        metrics.fleiss_kappa = kappa
        metrics.accuracy_stdev = compute_accuracy_stdev(per_repeat)

    return metrics
