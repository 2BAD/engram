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
    implementation: str,
    dataset: str,
    results: list[RunResult],
    sampling: dict | None = None,
    label: str | None = None,
) -> None:
    """Save experiment results to exp_dir/results.json."""
    data: dict = {
        'experiment_id': experiment_id,
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
    if label is not None:
        data['label'] = label.strip()
    (exp_dir / 'results.json').write_text(json.dumps(data, indent=2))


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
