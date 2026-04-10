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
    # True when precision/recall/F1 were computed per-class via macro averaging (enum
    # field + exact_match scorer). False for numeric/fuzzy/custom fields, where those
    # three values fall back to accuracy and display should render them as "—".
    is_classification: bool = False
    # Repeat-aware metrics, populated only when an experiment has multiple repeats per
    # input (engram eval --repeat N). All four are None for single-repeat runs.
    # mean_agreement_rate: per-input fraction of repeats matching the modal answer, averaged across inputs.
    # majority_rate: fraction of inputs where strictly more than N/2 repeats agreed; only defined for N >= 3.
    # fleiss_kappa: chance-corrected inter-repeat agreement (Fleiss 1971); only meaningful for categorical fields.
    # accuracy_stdev: stdev of per-repeat field accuracies; the noise floor used by Tier 3 baseline gating.
    mean_agreement_rate: float | None = None
    majority_rate: float | None = None
    fleiss_kappa: float | None = None
    accuracy_stdev: float | None = None


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
    matched_examples: int = 0
    field_metrics: list[FieldMetrics] = field(default_factory=list)
    confusion_matrices: list[ConfusionMatrix] = field(default_factory=list)
    cost_total_usd: float = 0.0
    cost_avg_usd: float = 0.0
    cost_median_usd: float = 0.0
    cost_p95_usd: float = 0.0
