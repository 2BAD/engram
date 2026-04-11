"""Experiment index: JSONL log of scored experiments, keyed by experiment id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.eval.results import load_results

if TYPE_CHECKING:
    from engram.models.scoring import EvalReport


def append_to_index(root: Path, report: EvalReport) -> None:
    """Upsert an experiment summary into experiments.jsonl, replacing any prior entry with the same id."""
    exp_dir = root / 'experiments' / report.experiment_id
    metadata, results = load_results(exp_dir)

    # Load config snapshot for model info
    snapshot_path = exp_dir / 'config-snapshot.json'
    models: list[str] = []
    if snapshot_path.exists():
        snap = json.loads(snapshot_path.read_text())
        models = snap.get('models', [])

    macro_accuracy = (
        sum(fm.accuracy for fm in report.field_metrics) / len(report.field_metrics) if report.field_metrics else 0.0
    )

    macro_f1 = sum(fm.f1 for fm in report.field_metrics) / len(report.field_metrics) if report.field_metrics else 0.0

    summary = {
        'short_id': metadata['short_id'],
        'id': report.experiment_id,
        'implementation': metadata['implementation'],
        'dataset': metadata['dataset'],
        'timestamp': metadata['timestamp'],
        'models': models,
        'matched_examples': report.matched_examples,
        'macro_accuracy': round(macro_accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'field_accuracy': {fm.field_name: round(fm.accuracy, 4) for fm in report.field_metrics},
        'field_precision': {fm.field_name: round(fm.precision, 4) for fm in report.field_metrics},
        'field_recall': {fm.field_name: round(fm.recall, 4) for fm in report.field_metrics},
        'field_f1': {fm.field_name: round(fm.f1, 4) for fm in report.field_metrics},
        'cost': {
            'total_usd': round(report.cost_total_usd, 4),
            'avg_usd': round(report.cost_avg_usd, 4),
        },
    }

    # Repeat-aware metrics. Persist only when at least one field carries them so single-repeat
    # entries keep the schema they had before Tier 2. The walrus filters narrow the optional types
    # for the type checker; they also serve as defensive runtime guards.
    repeat_aware_fields = [fm for fm in report.field_metrics if fm.accuracy_stdev is not None]
    if repeat_aware_fields:
        summary['repeats'] = max(r.repeat_index for r in results) + 1
        summary['field_accuracy_stdev'] = {
            fm.field_name: round(stdev, 4)
            for fm in repeat_aware_fields
            if (stdev := fm.accuracy_stdev) is not None
        }
        summary['field_mean_agreement_rate'] = {
            fm.field_name: round(value, 4)
            for fm in repeat_aware_fields
            if (value := fm.mean_agreement_rate) is not None
        }
        summary['field_majority_rate'] = {
            fm.field_name: round(value, 4)
            for fm in repeat_aware_fields
            if (value := fm.majority_rate) is not None
        }
        summary['field_fleiss_kappa'] = {
            fm.field_name: round(value, 4)
            for fm in repeat_aware_fields
            if (value := fm.fleiss_kappa) is not None
        }

    # Calibration data for `engram estimate`: mean output tokens across runs
    # where we actually got a response. Omitted when absent so the estimator
    # falls back to its default instead of reading a degenerate 0.
    output_tokens = [r.usage.completion_tokens for r in results if r.usage.completion_tokens > 0]
    if output_tokens:
        summary['avg_output_tokens'] = round(sum(output_tokens) / len(output_tokens))

    index_path = root / 'experiments' / 'experiments.jsonl'
    existing = read_index(root)
    replaced = False
    for i, entry in enumerate(existing):
        if entry.get('id') == summary['id']:
            existing[i] = summary
            replaced = True
            break
    if not replaced:
        existing.append(summary)

    with index_path.open('w') as f:
        for entry in existing:
            f.write(json.dumps(entry) + '\n')


def read_index(root: Path) -> list[dict[str, Any]]:
    """Read all entries from the experiment index."""
    index_path = root / 'experiments' / 'experiments.jsonl'
    if not index_path.exists():
        return []
    entries = []
    for line in index_path.read_text().strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def resolve_experiment_id(root: Path, arg: str) -> str:
    """
    Resolve a user-provided experiment reference to a full experiment id.

    A purely numeric ``arg`` is treated as a ``short_id`` lookup: the index is
    checked first (scored experiments), then every ``experiments/*/results.json``
    as a fallback so runs that haven't been scored yet are still reachable by
    number. Anything else is returned unchanged.
    """
    if not arg.isdigit():
        return arg
    target = int(arg)
    for entry in read_index(root):
        if entry.get('short_id') == target:
            return entry['id']
    experiments_dir = root / 'experiments'
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            results_file = exp_dir / 'results.json'
            if not results_file.exists():
                continue
            try:
                data = json.loads(results_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get('short_id') == target:
                return data['experiment_id']
    msg = f'No experiment found with short_id #{target}'
    raise FileNotFoundError(msg)
