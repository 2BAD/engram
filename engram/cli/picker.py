"""Interactive experiment picker for commands that accept an experiment ID."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import IntPrompt

from engram.tracking.index import read_index

console = Console()

_DEFAULT_LIMIT = 10


def _is_interactive() -> bool:
    """Whether stdin is a TTY. Factored out so tests can patch it without fighting CliRunner."""
    return sys.stdin.isatty()


def pick_experiment_id(root: Path, limit: int = _DEFAULT_LIMIT) -> str:
    """Prompt the user to pick an experiment by number; exits 1 if stdin isn't a TTY or the index is empty."""
    if not _is_interactive():
        console.print('[red]No experiment ID provided and stdin is not interactive.[/red]')
        console.print(
            'Hint: pass the experiment ID explicitly, or run engram from a terminal to pick from a list.'
        )
        raise typer.Exit(1)

    entries = read_index(root)
    if not entries:
        console.print('[red]No experiments available to pick from.[/red]')
        console.print(
            'Run [cyan]engram eval <impl> --dataset <name>[/cyan] and '
            '[cyan]engram score <id> --save[/cyan] to populate the index first.'
        )
        raise typer.Exit(1)

    entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    entries = entries[:limit]

    console.print('[bold]Recent experiments:[/bold]')
    for i, entry in enumerate(entries, start=1):
        ts = _format_timestamp(entry.get('timestamp', ''))
        accuracy = entry.get('macro_accuracy')
        acc_str = f'{accuracy:.1%}' if accuracy is not None else '—'
        console.print(
            f'  [bold cyan]{i:>2}[/bold cyan]  {entry["id"]}  [dim]{ts}  acc {acc_str}[/dim]'
        )
    console.print()

    choice = IntPrompt.ask(
        'Pick an experiment',
        choices=[str(i) for i in range(1, len(entries) + 1)],
        show_choices=False,
    )
    return entries[int(choice) - 1]['id']


def _format_timestamp(raw: str) -> str:
    if not raw:
        return ''
    try:
        return datetime.fromisoformat(raw).strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return raw
