"""Interactive prompts for CLI commands using questionary."""

from __future__ import annotations

import sys
from pathlib import Path

import questionary
from questionary import Choice

from engram.config.discovery import discover_datasets, discover_implementations
from engram.display.experiment_ref import format_ref_medium, format_when
from engram.tracking.index import decorate_with_short_ids, list_experiments


def is_interactive() -> bool:
    """Whether stdin is a TTY. Factored out so tests can patch it."""
    return sys.stdin.isatty()


def ask_implementation(root: Path) -> str:
    """Arrow-key select from discovered implementations. Auto-picks when there is only one."""
    impls = discover_implementations(root)
    if not impls:
        raise SystemExit('No implementations found. Run `engram init` to scaffold a project.')
    if len(impls) == 1:
        return impls[0]
    return questionary.select('Select implementation:', choices=impls).unsafe_ask()


def ask_dataset(root: Path) -> str:
    """Arrow-key select from discovered datasets. Auto-picks when there is only one."""
    datasets = discover_datasets(root)
    if not datasets:
        raise SystemExit('No datasets found. Add a dataset directory under datasets/.')
    if len(datasets) == 1:
        return datasets[0]
    return questionary.select('Select dataset:', choices=datasets).unsafe_ask()


def ask_experiment(root: Path, limit: int = 10) -> str:
    """Arrow-key select from recent experiments. Returns the full experiment id."""
    entries = list_experiments(root)
    if not entries:
        raise SystemExit('No experiments found. Run `engram run <impl> --dataset <name>` first.')

    entries = entries[:limit]
    decorate_with_short_ids(entries, root)
    choices = []
    for entry in entries:
        ref = format_ref_medium(entry)
        when = format_when(entry.get('timestamp', ''))
        accuracy = entry.get('macro_accuracy')
        acc_str = f'acc {accuracy:.1%}' if accuracy is not None else ''
        label = f'{ref}  {when}  {acc_str}'.rstrip()
        choices.append(Choice(title=label, value=entry.get('experiment_id', entry.get('id', ''))))

    return questionary.select('Select experiment:', choices=choices).unsafe_ask()


_PAIR_SIZE = 2


def ask_experiment_pair(root: Path, limit: int = 10) -> tuple[str, str]:
    """Checkbox select of exactly two experiments. Returns (older, newer) by timestamp."""
    entries = list_experiments(root)
    if not entries:
        raise SystemExit('No experiments found. Run `engram run <impl> --dataset <name>` first.')
    if len(entries) < _PAIR_SIZE:
        raise SystemExit('Need at least two experiments to compare.')

    entries = entries[:limit]
    decorate_with_short_ids(entries, root)
    choices = []
    for entry in entries:
        ref = format_ref_medium(entry)
        when = format_when(entry.get('timestamp', ''))
        accuracy = entry.get('macro_accuracy')
        acc_str = f'acc {accuracy:.1%}' if accuracy is not None else ''
        label = f'{ref}  {when}  {acc_str}'.rstrip()
        choices.append(Choice(title=label, value=entry))

    selected = questionary.checkbox(
        'Select two experiments to compare:',
        choices=choices,
        validate=lambda sel: len(sel) == _PAIR_SIZE or 'Select exactly 2 experiments',
    ).unsafe_ask()

    selected.sort(key=lambda e: e.get('timestamp', ''))
    return (
        selected[0].get('experiment_id', selected[0].get('id', '')),
        selected[1].get('experiment_id', selected[1].get('id', '')),
    )


def ask_label() -> str | None:
    """Prompt for an optional experiment label. Returns None on empty input."""
    answer = questionary.text('Label (optional):').unsafe_ask()
    return answer.strip() or None


def ask_confirm(message: str, default: bool = False) -> bool:
    """Yes/no confirmation prompt."""
    return questionary.confirm(message, default=default).unsafe_ask()
