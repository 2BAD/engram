"""Status command: show project overview."""

import typer
from rich.console import Console

from engram.config.discovery import (
    discover_datasets,
    discover_implementations,
    discover_workflows,
    find_project_root,
)
from engram.config.loader import load_implementation, load_project
from engram.config.validation import validate_project
from engram.tracking.baseline import load_baselines

console = Console()


def status_command() -> None:
    """Show project overview: workflows, implementations, datasets."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found in current or parent directories.[/red]')
        raise typer.Exit(1)

    project = load_project(root)
    console.print(f'[bold]Project:[/bold] {project.name}')
    console.print()

    workflows = discover_workflows(root)
    implementations = discover_implementations(root)
    datasets = discover_datasets(root)
    baselines = load_baselines(root)

    if workflows:
        console.print('[bold]Workflows:[/bold]')
        for name in workflows:
            baseline = baselines.get(name, {}).get('baseline')
            if baseline:
                console.print(f'  {name} [dim](baseline: {baseline})[/dim]')
            else:
                console.print(f'  {name}')
    else:
        console.print('[dim]Workflows: (none)[/dim]')

    if implementations:
        console.print('[bold]Implementations:[/bold]')
        for name in implementations:
            try:
                impl = load_implementation(root, name)
                reference = baselines.get(impl.workflow, {}).get('references', {}).get(name)
                suffix = f' [dim](ref: {reference})[/dim]' if reference else ''
                console.print(f'  {name} ({impl.platform}/{impl.runner}){suffix}')
            except (OSError, KeyError):
                console.print(f'  {name} (error loading)')
    else:
        console.print('[dim]Implementations: (none)[/dim]')

    _print_list('Datasets', datasets)

    errors = validate_project(root)
    if errors:
        console.print()
        console.print('[red bold]Validation errors:[/red bold]')
        for error in errors:
            console.print(f'  [red]{error}[/red]')


def _print_list(title: str, items: list[str]) -> None:
    if items:
        console.print(f'[bold]{title}:[/bold]')
        for item in items:
            console.print(f'  {item}')
    else:
        console.print(f'[dim]{title}: (none)[/dim]')
