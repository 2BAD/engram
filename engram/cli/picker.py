"""Interactive experiment picker for commands that accept an experiment ID."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import IntPrompt

from engram.display.experiment_ref import format_ref_long, format_ref_medium, format_when
from engram.tracking.index import read_index, resolve_experiment_id

console = Console()

_DEFAULT_LIMIT = 10


def _is_interactive() -> bool:
    """Whether stdin is a TTY. Factored out so tests can patch it without fighting CliRunner."""
    return sys.stdin.isatty()


def resolve_experiment_arg(
    root: Path,
    arg: str,
    impl: str | None = None,
    dataset: str | None = None,
) -> str:
    """
    Resolve a command-line experiment argument to a full experiment id.

    Wraps :func:`engram.tracking.index.resolve_experiment_id` with the CLI
    error contract: on ``FileNotFoundError``, print a red message and exit 1
    instead of letting the traceback bubble up. When the input was shortened
    (``@``, ``@~N``, or a short id) the resolved full id is echoed in dim
    style so lookups are never silent; a full-id pass-through stays quiet.
    """
    try:
        resolved = resolve_experiment_id(root, arg, impl=impl, dataset=dataset)
    except FileNotFoundError as e:
        console.print(f'[red]{e}[/red]')
        raise typer.Exit(1) from None
    if resolved != arg:
        pretty = _pretty_for_echo(root, resolved)
        console.print(f'[dim]resolved {arg} → {pretty}[/dim]')
    return resolved


def _pretty_for_echo(root: Path, experiment_id: str) -> str:
    """Format an experiment id as ``#N impl/dataset YYYY-MM-DD HH:MM`` for the resolver echo."""
    results_path = root / 'experiments' / experiment_id / 'results.json'
    if not results_path.exists():
        return experiment_id
    try:
        metadata = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return experiment_id
    return format_ref_long(metadata)


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
            'Run [cyan]engram run <impl> --dataset <name>[/cyan] and '
            '[cyan]engram score <id> --save[/cyan] to populate the index first.'
        )
        raise typer.Exit(1)

    entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    entries = entries[:limit]

    console.print('[bold]Recent experiments:[/bold]')
    for i, entry in enumerate(entries, start=1):
        when = format_when(entry.get('timestamp', ''))
        accuracy = entry.get('macro_accuracy')
        acc_str = f'{accuracy:.1%}' if accuracy is not None else '—'
        console.print(
            f'  [bold cyan]{i:>2}[/bold cyan]  {format_ref_medium(entry)}  [dim]{when}  acc {acc_str}[/dim]'
        )
    console.print()

    choice = IntPrompt.ask(
        'Pick an experiment',
        choices=[str(i) for i in range(1, len(entries) + 1)],
        show_choices=False,
    )
    return entries[int(choice) - 1]['id']
