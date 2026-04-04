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

    _print_list('Workflows', workflows)

    if implementations:
        lines = []
        for name in implementations:
            try:
                impl = load_implementation(root, name)
                lines.append(f'{name} ({impl.platform}/{impl.runner})')
            except (OSError, KeyError):
                lines.append(f'{name} (error loading)')
        console.print(f'[bold]Implementations:[/bold] {", ".join(lines)}')
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
        console.print(f'[bold]{title}:[/bold] {", ".join(items)}')
    else:
        console.print(f'[dim]{title}: (none)[/dim]')
