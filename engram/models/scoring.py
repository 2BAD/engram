"""Scoring and evaluation report models."""

from dataclasses import dataclass, field


@dataclass
class FieldMetrics:
    """Aggregate metrics for a single output field."""

    field_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total: int = 0
    correct: int = 0


@dataclass
class ConfusionMatrix:
    """Confusion matrix for a single field."""

    field_name: str
    labels: list[str] = field(default_factory=list)
    matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Full evaluation report for a scored experiment."""

    experiment_id: str
    field_metrics: list[FieldMetrics] = field(default_factory=list)
    confusion_matrices: list[ConfusionMatrix] = field(default_factory=list)
    cost_total_usd: float = 0.0
    cost_avg_usd: float = 0.0
    cost_median_usd: float = 0.0
    cost_p95_usd: float = 0.0
