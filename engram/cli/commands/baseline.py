"""Baseline command: manage workflow baselines and implementation references."""

import json
from typing import Annotated

import typer
from rich.console import Console

from engram.cli.picker import pick_experiment_id
from engram.config.discovery import find_project_root
from engram.display.baseline import print_baseline_status
from engram.observability.output_mode import get_output_mode
from engram.tracking.baseline import (
    load_baselines,
    lookup_experiment,
    set_impl_reference,
    set_workflow_baseline,
)

console = Console()

baseline_app = typer.Typer(
    name='baseline',
    help='Manage workflow baselines and implementation references.',
    no_args_is_help=True,
)


def _resolve(experiment_id: str | None) -> tuple:
    """Find the project root and resolve the experiment's (workflow, implementation); picker runs when `experiment_id` is None."""  # noqa: E501
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    if experiment_id is None:
        experiment_id = pick_experiment_id(root)

    try:
        workflow, implementation = lookup_experiment(root, experiment_id)
    except FileNotFoundError as e:
        console.print(f'[red]{e}[/red]')
        raise typer.Exit(1) from None
    except (KeyError, OSError) as e:
        console.print(f'[red]Could not resolve experiment {experiment_id}: {e}[/red]')
        raise typer.Exit(1) from None

    return root, workflow, implementation, experiment_id


@baseline_app.command('set')
def set_baseline(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help='Experiment ID to mark as the workflow baseline. Omit to pick interactively.'),
    ] = None,
) -> None:
    """Set this experiment as the workflow baseline (the frozen anchor)."""
    root, workflow, _impl, experiment_id = _resolve(experiment_id)
    set_workflow_baseline(root, workflow, experiment_id)
    console.print(f'[green]Set baseline for workflow [bold]{workflow}[/bold]: {experiment_id}[/green]')


@baseline_app.command('promote')
def promote_reference(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help='Experiment ID to promote as its implementation reference. Omit to pick interactively.'),
    ] = None,
) -> None:
    """Promote this experiment to be its implementation's current reference."""
    root, workflow, implementation, experiment_id = _resolve(experiment_id)
    set_impl_reference(root, workflow, implementation, experiment_id)
    console.print(
        f'[green]Promoted [bold]{implementation}[/bold] reference for workflow '
        f'[bold]{workflow}[/bold]: {experiment_id}[/green]'
    )


@baseline_app.command('show')
def show_baselines() -> None:
    """Show current workflow baselines and implementation references."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    baselines = load_baselines(root)
    if get_output_mode().use_rich:
        print_baseline_status(baselines)
    else:
        print(json.dumps(baselines, indent=2))
