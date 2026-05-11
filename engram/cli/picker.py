"""Interactive experiment picker for commands that accept an experiment ID."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from engram.cli.prompts import ask_confirm, ask_experiment, ask_experiment_pair, is_interactive
from engram.display.experiment_ref import format_ref_long, linkify_ref
from engram.tracking.index import decorate_with_short_ids, resolve_experiment_id

console = Console()


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
    """Format an experiment id as a linked ``#N impl/dataset YYYY-MM-DD HH:MM`` for the resolver echo."""
    results_path = root / 'experiments' / experiment_id / 'results.json'
    if not results_path.exists():
        return experiment_id
    try:
        metadata = json.loads(results_path.read_text())
    except json.JSONDecodeError, OSError:
        return experiment_id
    decorate_with_short_ids([metadata], root)
    return linkify_ref(format_ref_long(metadata), root / 'experiments' / experiment_id)


def pick_experiment_id(root: Path, limit: int = 100) -> str:
    """Prompt the user to pick an experiment with arrow-key navigation; exits 1 if stdin isn't a TTY."""
    if not is_interactive():
        console.print('[red]No experiment ID provided and stdin is not interactive.[/red]')
        console.print('Hint: pass the experiment ID explicitly, or run engram from a terminal to pick from a list.')
        raise typer.Exit(1)

    try:
        return ask_experiment(root, limit=limit)
    except SystemExit as e:
        console.print(f'[red]{e}[/red]')
        raise typer.Exit(1) from None


def pick_one_or_pair(root: Path, limit: int = 100) -> tuple[str, str | None]:
    """Ask the user whether to analyze one or two experiments, then pick accordingly. Returns (id_a, id_b or None)."""
    if not is_interactive():
        console.print('[red]No experiment ID provided and stdin is not interactive.[/red]')
        console.print('Hint: pass experiment IDs explicitly, or run engram from a terminal to pick from a list.')
        raise typer.Exit(1)

    try:
        if ask_confirm('Compare two experiments?', default=False):
            a, b = ask_experiment_pair(root, limit=limit)
            return a, b
        return ask_experiment(root, limit=limit), None
    except SystemExit as e:
        console.print(f'[red]{e}[/red]')
        raise typer.Exit(1) from None


def pick_experiment_pair(root: Path, limit: int = 100) -> tuple[str, str]:
    """Prompt the user to pick two experiments with checkbox navigation; exits 1 if stdin isn't a TTY."""
    if not is_interactive():
        console.print('[red]No experiment IDs provided and stdin is not interactive.[/red]')
        console.print('Hint: pass two experiment IDs explicitly, or run engram from a terminal to pick from a list.')
        raise typer.Exit(1)

    try:
        return ask_experiment_pair(root, limit=limit)
    except SystemExit as e:
        console.print(f'[red]{e}[/red]')
        raise typer.Exit(1) from None
