"""Shell completion callbacks for CLI parameters."""

from __future__ import annotations

from engram.config.discovery import (
    discover_datasets,
    discover_implementations,
    find_project_root,
)
from engram.tracking.index import decorate_with_short_ids, list_experiments


def complete_implementations(incomplete: str) -> list[tuple[str, str]]:
    """Complete implementation names from the project directory."""
    root = find_project_root()
    if root is None:
        return []
    return [(name, '') for name in discover_implementations(root) if name.startswith(incomplete)]


def complete_datasets(incomplete: str) -> list[tuple[str, str]]:
    """Complete dataset names from the project directory."""
    root = find_project_root()
    if root is None:
        return []
    return [(name, '') for name in discover_datasets(root) if name.startswith(incomplete)]


def complete_experiment_ids(incomplete: str) -> list[tuple[str, str]]:
    """
    Complete experiment IDs from the project index.

    Offers #N short IDs (with impl/dataset as help text), @ for the most
    recent experiment, and @~N for older ones.
    """
    root = find_project_root()
    if root is None:
        return []

    experiments = list_experiments(root)
    if not experiments:
        return []
    decorate_with_short_ids(experiments, root)

    completions: list[tuple[str, str]] = []

    # @ and @~N references
    if not incomplete or incomplete.startswith('@'):
        completions.append(('@', 'most recent'))
        for i in range(1, min(len(experiments), 10)):
            ref = f'@~{i}'
            exp = experiments[i]
            hint = f'{exp.get("implementation", "")}/{exp.get("dataset", "")}'
            completions.append((ref, hint))
        completions = [(ref, hint) for ref, hint in completions if ref.startswith(incomplete)]

    # #N short IDs
    if not incomplete or incomplete.startswith('#'):
        for exp in experiments:
            short_id = exp.get('short_id')
            if short_id is None:
                continue
            ref = f'#{short_id}'
            hint = f'{exp.get("implementation", "")}/{exp.get("dataset", "")}'
            if ref.startswith(incomplete):
                completions.append((ref, hint))

    return completions
