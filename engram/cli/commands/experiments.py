"""Experiments command: list recent experiments from the index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from engram.cli.completions import complete_datasets, complete_implementations
from engram.config.discovery import find_project_root
from engram.datasets.loader import compute_labels_hash, load_dataset_labels
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
        typer.Option('--impl', '-i', help='Filter by implementation name.', autocompletion=complete_implementations),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option('--dataset', '-d', help='Filter by dataset name.', autocompletion=complete_datasets),
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

    _annotate_labels_stale(root, entries)

    if get_output_mode().use_rich:
        _print_table(entries, total=total, truncated=truncated)
    else:
        print(json.dumps(entries, indent=2))


def _annotate_labels_stale(root: Path, entries: list[dict[str, Any]]) -> None:
    """Tag each entry with ``labels_stale=True`` when the current dataset hash no longer matches the stored one."""
    hash_cache: dict[str, str | None] = {}
    for entry in entries:
        stored = entry.get('labels_hash')
        if not stored:
            continue
        ds = entry.get('dataset', '')
        if ds not in hash_cache:
            hash_cache[ds] = _current_labels_hash(root, ds)
        current = hash_cache[ds]
        if current and current != stored:
            entry['labels_stale'] = True


def _current_labels_hash(root: Path, dataset: str) -> str | None:
    """Hash the current labels.json for a dataset. None when the file is absent or unreadable."""
    if not (root / 'datasets' / dataset / 'labels.json').exists():
        return None
    try:
        labels = load_dataset_labels(root, dataset)
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return None
    return compute_labels_hash(labels)


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

    has_labels = any(entry.get('label') for entry in entries)

    table = Table(title='Experiments')
    table.add_column(Text('#', justify='center'), style='bold cyan', justify='right')
    table.add_column(Text('When', justify='center'), justify='right')
    table.add_column(Text('Impl', justify='center'), overflow='fold')
    table.add_column(Text('Dataset', justify='center'), overflow='fold')
    if has_labels:
        table.add_column(Text('Label', justify='center'), overflow='fold')
    table.add_column(Text('Acc', justify='center'), justify='right')
    table.add_column(Text('F1', justify='center'), justify='right')
    table.add_column(Text('Cost', justify='center'), justify='right')
    table.add_column(Text('N', justify='center'), justify='right')

    any_stale = False
    for entry in entries:
        short_id = entry.get('short_id')
        stale = bool(entry.get('labels_stale'))
        any_stale = any_stale or stale
        dataset_cell = entry.get('dataset', '')
        if stale:
            dataset_cell = f'[yellow]{dataset_cell} *[/yellow]'
        row = [
            str(short_id) if short_id is not None else '[dim]—[/dim]',
            format_when(entry.get('timestamp', '')),
            entry.get('implementation', ''),
            dataset_cell,
        ]
        if has_labels:
            row.append(entry.get('label', '') or '[dim]—[/dim]')
        row.extend(
            [
                _format_pct(entry.get('macro_accuracy')),
                _format_pct(entry.get('macro_f1')),
                _format_cost(entry.get('cost', {}).get('total_usd')),
                str(entry.get('matched_examples', '')),
            ]
        )
        table.add_row(*row)

    console.print(table)

    shown = len(entries)
    if truncated:
        console.print(f'[dim]Showing {shown} of {total} experiments. Pass [bold]--limit 0[/bold] to show all.[/dim]')
    else:
        console.print(f'[dim]{shown} experiment(s).[/dim]')

    if any_stale:
        console.print(
            '[dim yellow]* labels have changed since this score — '
            're-run [bold]engram score <ref> --save[/bold] to refresh.[/dim yellow]'
        )


def _format_pct(value: float | None) -> str:
    if value is None:
        return '[dim]—[/dim]'
    return f'{value:.1%}'


def _format_cost(value: float | None) -> str:
    if value is None:
        return '[dim]—[/dim]'
    return f'${value:.4f}'
