"""Metrics aggregation: accuracy, precision, recall, F1, confusion matrices, cost stats."""

from __future__ import annotations

import statistics

from engram.models.scoring import ConfusionMatrix, FieldMetrics


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
