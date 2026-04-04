"""Metrics aggregation: accuracy, precision, recall, F1, confusion matrices, cost stats."""

from __future__ import annotations

import statistics

from engram.models.scoring import ConfusionMatrix, FieldMetrics


def compute_field_metrics(field_name: str, scores: list[bool]) -> FieldMetrics:
    """Compute accuracy, precision, recall, and F1 from a list of boolean scores."""
    total = len(scores)
    correct = sum(scores)
    accuracy = correct / total if total > 0 else 0.0

    # For binary correct/incorrect, precision = recall = accuracy
    # This is a simplification; per-class metrics require class-level data
    return FieldMetrics(
        field_name=field_name,
        accuracy=accuracy,
        precision=accuracy,
        recall=accuracy,
        f1=accuracy,
        total=total,
        correct=correct,
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
