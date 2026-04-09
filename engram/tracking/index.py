"""Experiment index: append-only JSONL log of scored experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.eval.results import load_results

if TYPE_CHECKING:
    from engram.models.scoring import EvalReport


def append_to_index(root: Path, report: EvalReport) -> None:
    """Append an experiment summary to experiments/experiments.jsonl."""
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

    summary = {
        'id': report.experiment_id,
        'implementation': metadata['implementation'],
        'dataset': metadata['dataset'],
        'timestamp': metadata['timestamp'],
        'models': models,
        'matched_examples': report.matched_examples,
        'macro_accuracy': round(macro_accuracy, 4),
        'field_accuracy': {fm.field_name: round(fm.accuracy, 4) for fm in report.field_metrics},
        'cost': {
            'total_usd': round(report.cost_total_usd, 4),
            'avg_usd': round(report.cost_avg_usd, 4),
        },
    }

    # Calibration data for `engram estimate`: mean output tokens across runs
    # where we actually got a response. Omitted when absent so the estimator
    # falls back to its default instead of reading a degenerate 0.
    output_tokens = [r.usage.completion_tokens for r in results if r.usage.completion_tokens > 0]
    if output_tokens:
        summary['avg_output_tokens'] = round(sum(output_tokens) / len(output_tokens))

    index_path = root / 'experiments' / 'experiments.jsonl'
    with index_path.open('a') as f:
        f.write(json.dumps(summary) + '\n')


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
