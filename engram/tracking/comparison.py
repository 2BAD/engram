"""Compare two experiments: accuracy deltas, cost differences, regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engram.scoring.engine import score_experiment


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
    """Accuracy delta for a single field."""

    field_name: str
    accuracy_a: float
    accuracy_b: float

    @property
    def delta(self) -> float:
        return self.accuracy_b - self.accuracy_a

    @property
    def regressed(self) -> bool:
        return self.delta < 0


def compare_experiments(root: Path, id_a: str, id_b: str) -> ComparisonResult:
    """Compare two experiments by scoring both and computing deltas."""
    report_a = score_experiment(root, id_a)
    report_b = score_experiment(root, id_b)

    field_deltas = {}
    metrics_a = {fm.field_name: fm for fm in report_a.field_metrics}
    metrics_b = {fm.field_name: fm for fm in report_b.field_metrics}

    all_fields = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    for field_name in all_fields:
        acc_a = metrics_a[field_name].accuracy if field_name in metrics_a else 0.0
        acc_b = metrics_b[field_name].accuracy if field_name in metrics_b else 0.0
        field_deltas[field_name] = FieldDelta(field_name=field_name, accuracy_a=acc_a, accuracy_b=acc_b)

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
