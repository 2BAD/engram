"""Metrics aggregation: accuracy, precision, recall, F1, confusion matrices, cost stats."""

from __future__ import annotations

import statistics
from collections import Counter

from engram.models.scoring import ConfusionMatrix, FieldMetrics

# Minimum repeats required for each repeat-aware metric to be defined.
# Mean agreement, Fleiss kappa, and accuracy stdev all need at least 2 repeats.
# Strict majority (> N/2) only becomes a non-trivial concept at N >= 3.
_MIN_REPEATS_FOR_AGREEMENT = 2
_MIN_REPEATS_FOR_MAJORITY = 3


def compute_field_metrics(
    field_name: str,
    scores: list[bool],
    pairs: list[tuple[str, str]] | None = None,
    is_classification: bool = False,
) -> FieldMetrics:
    """Compute accuracy plus macro-averaged P/R/F1 per class; non-classification fields fall back to accuracy."""
    total = len(scores)
    correct = sum(scores)
    accuracy = correct / total if total > 0 else 0.0

    if is_classification and pairs:
        precision, recall, f1 = _macro_classification_metrics(pairs)
    else:
        precision = recall = f1 = accuracy

    return FieldMetrics(
        field_name=field_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        total=total,
        correct=correct,
        is_classification=is_classification and bool(pairs),
    )


def _macro_classification_metrics(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Return macro-averaged (precision, recall, F1) over the union of classes in pairs."""
    labels = sorted({label for pair in pairs for label in pair})
    if not labels:
        return 0.0, 0.0, 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in labels:
        tp = sum(1 for expected, predicted in pairs if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in pairs if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in pairs if expected == label and predicted != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
    )


def compute_confusion_matrix(field_name: str, pairs: list[tuple[str, str]]) -> ConfusionMatrix:
    """Build a confusion matrix from (expected, predicted) pairs."""
    all_labels = sorted({label for pair in pairs for label in pair})

    matrix: dict[str, dict[str, int]] = {label: dict.fromkeys(all_labels, 0) for label in all_labels}

    for expected, predicted in pairs:
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1

    return ConfusionMatrix(
        field_name=field_name,
        labels=all_labels,
        matrix=matrix,
    )


def compute_agreement_metrics(
    per_input_predictions: dict[str, list[str]],
    repeats: int,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute (mean_agreement_rate, majority_rate, fleiss_kappa) for one field across N repeats.

    `per_input_predictions` maps each input filename to the list of stringified predictions
    that field received from successful repeats. Failed repeats are absent from the list,
    so per-input list lengths can be < repeats when partial failures occurred.

    All three metrics return None when there are fewer than 2 repeats configured or no
    successful predictions to score. `majority_rate` returns None unless `repeats >= 3`,
    where the strict-majority concept (>N/2) is meaningful. `fleiss_kappa` only counts
    items that have all `repeats` predictions present, since the standard Fleiss formula
    requires equal raters per item.
    """
    if repeats < _MIN_REPEATS_FOR_AGREEMENT or not per_input_predictions:
        return None, None, None

    mean_agreement = _mean_agreement_rate(per_input_predictions)
    majority = _majority_rate(per_input_predictions) if repeats >= _MIN_REPEATS_FOR_MAJORITY else None
    kappa = _fleiss_kappa(per_input_predictions, repeats)
    return mean_agreement, majority, kappa


def compute_accuracy_stdev(per_repeat_scores: dict[int, list[bool]]) -> float | None:
    """Stdev of per-repeat field accuracies, or None when fewer than 2 repeats produced any scores."""
    per_repeat_acc = [sum(scores) / len(scores) for scores in per_repeat_scores.values() if scores]
    if len(per_repeat_acc) < _MIN_REPEATS_FOR_AGREEMENT:
        return None
    return statistics.stdev(per_repeat_acc)


def _mean_agreement_rate(per_input_predictions: dict[str, list[str]]) -> float | None:
    """Per-input fraction of repeats matching the modal answer, averaged across inputs."""
    rates = []
    for predictions in per_input_predictions.values():
        if not predictions:
            continue
        modal_count = Counter(predictions).most_common(1)[0][1]
        rates.append(modal_count / len(predictions))
    return sum(rates) / len(rates) if rates else None


def _majority_rate(per_input_predictions: dict[str, list[str]]) -> float | None:
    """Fraction of inputs where strictly more than half the successful repeats agreed."""
    has_majority = []
    for predictions in per_input_predictions.values():
        if len(predictions) < _MIN_REPEATS_FOR_AGREEMENT:
            continue  # need at least 2 successful repeats to define majority
        modal_count = Counter(predictions).most_common(1)[0][1]
        has_majority.append(modal_count * 2 > len(predictions))
    return sum(has_majority) / len(has_majority) if has_majority else None


def _fleiss_kappa(per_input_predictions: dict[str, list[str]], repeats: int) -> float | None:
    """Chance-corrected agreement (Fleiss 1971) over items with all `repeats` predictions present."""
    rated_items = [preds for preds in per_input_predictions.values() if len(preds) == repeats]
    if len(rated_items) < _MIN_REPEATS_FOR_AGREEMENT or repeats < _MIN_REPEATS_FOR_AGREEMENT:
        return None

    all_labels = sorted({label for preds in rated_items for label in preds})
    if len(all_labels) < _MIN_REPEATS_FOR_AGREEMENT:
        # All items collapsed to the same single label. Agreement is trivially perfect.
        return 1.0

    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    n_items = len(rated_items)
    n_categories = len(all_labels)

    # table[i][j] = count of repeats that assigned item i to category j
    table = [[0] * n_categories for _ in range(n_items)]
    for i, preds in enumerate(rated_items):
        for label in preds:
            table[i][label_to_idx[label]] += 1

    # P_i = (sum_j table[i][j]^2 - repeats) / (repeats * (repeats - 1))
    p_bar = sum(
        (sum(c * c for c in row) - repeats) / (repeats * (repeats - 1)) for row in table
    ) / n_items

    # p_j = total assignments to category j / total assignments overall
    total_assignments = n_items * repeats
    p_j_values = [sum(row[j] for row in table) / total_assignments for j in range(n_categories)]
    p_e = sum(p * p for p in p_j_values)

    if p_e >= 1.0:
        # Degenerate: chance agreement is perfect, kappa is mathematically undefined.
        # Convention: report 1.0 since observed agreement is also perfect.
        return 1.0

    return (p_bar - p_e) / (1.0 - p_e)


def compute_cost_stats(costs: list[float]) -> tuple[float, float, float, float]:
    """
    Compute total, average, median, and p95 cost.

    Returns (total, avg, median, p95). All zeros if no costs.
    """
    if not costs:
        return 0.0, 0.0, 0.0, 0.0

    total = sum(costs)
    avg = total / len(costs)
    median = statistics.median(costs)

    sorted_costs = sorted(costs)
    p95_idx = int(len(sorted_costs) * 0.95)
    p95 = sorted_costs[min(p95_idx, len(sorted_costs) - 1)]

    return total, avg, median, p95
