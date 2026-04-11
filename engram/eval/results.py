"""Experiment results persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from engram.models.run import RunResult, TokenUsage


def save_results(  # noqa: PLR0913 — metadata fields are flat by design
    exp_dir: Path,
    experiment_id: str,
    short_id: int,
    implementation: str,
    dataset: str,
    results: list[RunResult],
    sampling: dict | None = None,
) -> None:
    """Save experiment results to exp_dir/results.json."""
    data: dict = {
        'experiment_id': experiment_id,
        'short_id': short_id,
        'implementation': implementation,
        'dataset': dataset,
        'timestamp': datetime.now(UTC).isoformat(),
        'total': len(results),
        'succeeded': sum(1 for r in results if r.status == 'succeeded'),
        'failed': sum(1 for r in results if r.status != 'succeeded'),
        'results': [asdict(r) for r in results],
    }
    if sampling is not None:
        data['sampling'] = sampling
    (exp_dir / 'results.json').write_text(json.dumps(data, indent=2))


def next_short_id(root: Path) -> int:
    """
    Compute the next monotonic short_id for a new experiment in this project.

    Scans every ``experiments/*/results.json`` for the highest assigned short_id
    and returns max + 1. Returns 1 for a project with no prior experiments. Runs
    without a short_id (from an older schema) are skipped rather than renumbered.
    """
    experiments_dir = root / 'experiments'
    if not experiments_dir.exists():
        return 1
    max_id = 0
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
        sid = data.get('short_id')
        if isinstance(sid, int) and sid > max_id:
            max_id = sid
    return max_id + 1


def load_results(exp_dir: Path) -> tuple[dict, list[RunResult]]:
    """
    Load experiment results from exp_dir/results.json.

    Returns (metadata_dict, list_of_RunResult).
    """
    raw = json.loads((exp_dir / 'results.json').read_text())
    results = []
    for r in raw['results']:
        usage_data = r.pop('usage', {})
        usage = TokenUsage(**usage_data)
        results.append(RunResult(usage=usage, **r))
    metadata = {k: v for k, v in raw.items() if k != 'results'}
    return metadata, results
