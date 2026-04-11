"""Experiments command: list recent experiments from the index."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from engram.config.discovery import find_project_root
from engram.display.experiment_ref import format_when
from engram.observability.output_mode import get_output_mode
from engram.tracking.index import read_index

console = Console()

experiments_app = typer.Typer(
    name='experiments',
    help='Inspect the experiments tracked in the index.',
    no_args_is_help=True,
)


@experiments_app.command('list')
def list_experiments(
    limit: Annotated[
        int,
        typer.Option('--limit', '-n', help='Maximum rows to show. Pass 0 to show all.'),
    ] = 20,
    implementation: Annotated[
        str | None,
        typer.Option('--impl', '-i', help='Filter by implementation name.'),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option('--dataset', '-d', help='Filter by dataset name.'),
    ] = None,
) -> None:
    """List recent scored experiments, most recent first."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    entries = read_index(root)
    total = len(entries)

    if implementation is not None:
        entries = [e for e in entries if e.get('implementation') == implementation]
    if dataset is not None:
        entries = [e for e in entries if e.get('dataset') == dataset]

    # Most recent first. Entries without a timestamp sort last.
    entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)

    truncated = False
    if limit > 0 and len(entries) > limit:
        truncated = True
        entries = entries[:limit]

    if get_output_mode().use_rich:
        _print_table(entries, total=total, truncated=truncated)
    else:
        print(json.dumps(entries, indent=2))


def _print_table(entries: list[dict[str, Any]], total: int, truncated: bool) -> None:
    if not entries:
        if total == 0:
            console.print(
                '[dim]No experiments in the index yet. Run [bold]engram run <impl> --dataset <name>[/bold] '
                'and then [bold]engram score <experiment-id> --save[/bold] to populate it.[/dim]'
            )
        else:
            console.print('[dim]No experiments match the given filters.[/dim]')
        return

    table = Table(title='Experiments')
    table.add_column('#', style='bold cyan', justify='right')
    table.add_column('When', justify='right')
    table.add_column('Impl', overflow='fold')
    table.add_column('Dataset', overflow='fold')
    table.add_column('Acc', justify='right')
    table.add_column('F1', justify='right')
    table.add_column('Cost', justify='right')
    table.add_column('N', justify='right')

    for entry in entries:
        short_id = entry.get('short_id')
        table.add_row(
            str(short_id) if short_id is not None else '[dim]—[/dim]',
            format_when(entry.get('timestamp', '')),
            entry.get('implementation', ''),
            entry.get('dataset', ''),
            _format_pct(entry.get('macro_accuracy')),
            _format_pct(entry.get('macro_f1')),
            _format_cost(entry.get('cost', {}).get('total_usd')),
            str(entry.get('matched_examples', '')),
        )

    console.print(table)

    shown = len(entries)
    if truncated:
        console.print(f'[dim]Showing {shown} of {total} experiments. Pass [bold]--limit 0[/bold] to show all.[/dim]')
    else:
        console.print(f'[dim]{shown} experiment(s).[/dim]')


def _format_pct(value: float | None) -> str:
    if value is None:
        return '[dim]—[/dim]'
    return f'{value:.1%}'


def _format_cost(value: float | None) -> str:
    if value is None:
        return '[dim]—[/dim]'
    return f'${value:.4f}'
