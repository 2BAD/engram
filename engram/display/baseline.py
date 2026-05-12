"""Render baseline status grouped by workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from engram.display.experiment_ref import format_ref_long, linkify_ref
from engram.tracking.index import decorate_with_short_ids

console = Console()


def print_baseline_status(baselines: dict[str, dict[str, Any]], root: Path | None = None) -> None:
    """Print workflow baselines and per-implementation references as pretty refs."""
    if not baselines:
        console.print('[dim]No baselines set.[/dim]')
        console.print('  Set one with [bold]engram baseline set <experiment-id>[/bold]')
        return

    for workflow in sorted(baselines):
        entry = baselines[workflow]
        console.print(f'[bold]{workflow}[/bold]')

        baseline = entry.get('baseline')
        if baseline:
            console.print(f'  baseline:   {_pretty_ref(root, baseline)}')
        else:
            console.print('  [dim]baseline:   (none)[/dim]')

        references = entry.get('references', {})
        if references:
            console.print('  references:')
            width = max(len(name) for name in references)
            for impl in sorted(references):
                console.print(f'    {impl:<{width}}  {_pretty_ref(root, references[impl])}')


def _pretty_ref(root: Path | None, experiment_id: str) -> str:
    """Format an experiment id as ``#N impl/dataset YYYY-MM-DD HH:MM`` by reading its metadata."""
    if root is None:
        return experiment_id
    results_path = root / 'experiments' / experiment_id / 'results.json'
    if not results_path.exists():
        return f'[dim]{experiment_id} (missing)[/dim]'
    try:
        metadata = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return f'[dim]{experiment_id} (unreadable)[/dim]'
    decorate_with_short_ids([metadata], root)
    return linkify_ref(format_ref_long(metadata), root / 'experiments' / experiment_id)
