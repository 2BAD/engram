"""Render baseline status grouped by workflow."""

from __future__ import annotations

from typing import Any

from rich.console import Console

console = Console()


def print_baseline_status(baselines: dict[str, dict[str, Any]]) -> None:
    """Print workflow baselines and per-implementation references."""
    if not baselines:
        console.print('[dim]No baselines set.[/dim]')
        console.print('  Set one with [bold]engram baseline set <experiment-id>[/bold]')
        return

    for workflow in sorted(baselines):
        entry = baselines[workflow]
        console.print(f'[bold]{workflow}[/bold]')

        baseline = entry.get('baseline')
        if baseline:
            console.print(f'  baseline:   {baseline}')
        else:
            console.print('  [dim]baseline:   (none)[/dim]')

        references = entry.get('references', {})
        if references:
            console.print('  references:')
            width = max(len(name) for name in references)
            for impl in sorted(references):
                console.print(f'    {impl:<{width}}  {references[impl]}')
