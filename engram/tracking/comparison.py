"""Compare two experiments: accuracy deltas, cost differences, regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from engram.scoring.engine import load_saved_report, score_experiment

if TYPE_CHECKING:
    from engram.models.scoring import EvalReport


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""

    experiment_a: str
    experiment_b: str
    field_deltas: dict[str, FieldDelta] = field(default_factory=dict)
    cost_a: dict[str, float] = field(default_factory=dict)
    cost_b: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)


@dataclass
class FieldDelta:
    """Per-field metric deltas across two experiments. Regression is gated on F1, not accuracy."""

    field_name: str
    accuracy_a: float
    accuracy_b: float
    precision_a: float = 0.0
    precision_b: float = 0.0
    recall_a: float = 0.0
    recall_b: float = 0.0
    f1_a: float = 0.0
    f1_b: float = 0.0
    # True if either side reported real per-class metrics. If both sides are
    # non-classification, precision/recall/F1 values are fallback=accuracy and
    # display should render them as "—".
    is_classification: bool = False

    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_b - self.accuracy_a

    @property
    def precision_delta(self) -> float:
        return self.precision_b - self.precision_a

    @property
    def recall_delta(self) -> float:
        return self.recall_b - self.recall_a

    @property
    def f1_delta(self) -> float:
        return self.f1_b - self.f1_a

    @property
    def delta(self) -> float:
        """Primary metric delta used for regression detection (F1)."""
        return self.f1_delta

    @property
    def regressed(self) -> bool:
        return self.f1_delta < 0


def compare_experiments(
    root: Path,
    id_a: str,
    id_b: str,
    report_a: EvalReport | None = None,
    report_b: EvalReport | None = None,
) -> ComparisonResult:
    """Compare two experiments by scoring both and computing deltas."""
    if report_a is None:
        report_a = load_saved_report(root, id_a) or score_experiment(root, id_a)
    if report_b is None:
        report_b = load_saved_report(root, id_b) or score_experiment(root, id_b)

    field_deltas = {}
    metrics_a = {fm.field_name: fm for fm in report_a.field_metrics}
    metrics_b = {fm.field_name: fm for fm in report_b.field_metrics}

    all_fields = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    for field_name in all_fields:
        a = metrics_a.get(field_name)
        b = metrics_b.get(field_name)
        field_deltas[field_name] = FieldDelta(
            field_name=field_name,
            accuracy_a=a.accuracy if a else 0.0,
            accuracy_b=b.accuracy if b else 0.0,
            precision_a=a.precision if a else 0.0,
            precision_b=b.precision if b else 0.0,
            recall_a=a.recall if a else 0.0,
            recall_b=b.recall if b else 0.0,
            f1_a=a.f1 if a else 0.0,
            f1_b=b.f1 if b else 0.0,
            is_classification=(a is not None and a.is_classification) or (b is not None and b.is_classification),
        )

    regressions = [name for name, delta in field_deltas.items() if delta.regressed]

    return ComparisonResult(
        experiment_a=id_a,
        experiment_b=id_b,
        field_deltas=field_deltas,
        cost_a={'total': report_a.cost_total_usd, 'avg': report_a.cost_avg_usd},
        cost_b={'total': report_b.cost_total_usd, 'avg': report_b.cost_avg_usd},
        regressions=regressions,
    )


def diff_config_snapshots(root: Path, id_a: str, id_b: str, show_prompts: bool = False) -> list[str]:
    """Diff the config snapshots of two experiments. Returns a list of diff lines."""
    snap_a = _load_snapshot(root / 'experiments' / id_a)
    snap_b = _load_snapshot(root / 'experiments' / id_b)

    lines: list[str] = []

    # Model changes
    models_a = snap_a.get('models', [])
    models_b = snap_b.get('models', [])
    if models_a != models_b:
        lines.append(f'Models: {models_a} -> {models_b}')

    # Runner config changes
    rc_a = snap_a.get('runner_config', {})
    rc_b = snap_b.get('runner_config', {})
    for key in sorted(set(rc_a.keys()) | set(rc_b.keys())):
        if rc_a.get(key) != rc_b.get(key):
            lines.append(f'runner_config.{key}: {rc_a.get(key)!r} -> {rc_b.get(key)!r}')

    # Transform changes: diff even when only one side has one, so cross-impl
    # comparisons honestly flag that inputs/outputs were reshaped differently.
    tx_a = snap_a.get('transform', {})
    tx_b = snap_b.get('transform', {})
    for side in ('input', 'output'):
        if tx_a.get(side) != tx_b.get(side):
            lines.append(f'transform.{side}: {tx_a.get(side)!r} -> {tx_b.get(side)!r}')

    # Prompt changes
    prompts_a = snap_a.get('prompts', {})
    prompts_b = snap_b.get('prompts', {})
    for prompt_name in sorted(set(prompts_a.keys()) | set(prompts_b.keys())):
        text_a = prompts_a.get(prompt_name, '')
        text_b = prompts_b.get(prompt_name, '')
        if text_a != text_b:
            if show_prompts:
                import difflib  # noqa: PLC0415

                diff = difflib.unified_diff(
                    text_a.splitlines(keepends=True),
                    text_b.splitlines(keepends=True),
                    fromfile=f'{id_a}/{prompt_name}',
                    tofile=f'{id_b}/{prompt_name}',
                )
                lines.extend(diff)
            else:
                lines_a = len(text_a.splitlines())
                lines_b = len(text_b.splitlines())
                lines.append(f'Prompt {prompt_name}: changed ({lines_a} lines -> {lines_b} lines)')

    return lines


def _load_snapshot(exp_dir: Path) -> dict:
    """Load a config snapshot from an experiment directory."""
    path = exp_dir / 'config-snapshot.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text())
