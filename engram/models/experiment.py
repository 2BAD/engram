"""Experiment models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experiment:
    """A complete experiment: one implementation, one dataset, all results."""

    id: str
    implementation: str
    dataset: str
    timestamp: str
    config_snapshot_path: str
    results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentSummary:
    """Summary line for experiments.jsonl index."""

    id: str
    implementation: str
    dataset: str
    timestamp: str
    models: list[str] = field(default_factory=list)
    matched_examples: int = 0
    macro_accuracy: float = 0.0
    field_accuracy: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
