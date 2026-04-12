"""Score command: score experiment results against labels."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console

from engram.cli.picker import pick_experiment_id, resolve_experiment_arg
from engram.config.discovery import find_project_root
from engram.display.experiment_ref import format_ref_medium
from engram.display.tables import print_eval_report
from engram.eval.results import load_results
from engram.observability.output_mode import get_output_mode
from engram.scoring.engine import score_experiment
from engram.tracking.index import append_to_index

console = Console()


def score_command(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help='Experiment ID, short_id, or @ / @~N. Omit to pick interactively from recent runs.'),
    ] = None,
    save: Annotated[bool, typer.Option('--save', help='Save report and update experiment index')] = False,
    implementation: Annotated[
        str | None,
        typer.Option('--impl', '-i', help='Scope @ / @~N resolution to this implementation'),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option('--dataset', '-d', help='Scope @ / @~N resolution to this dataset'),
    ] = None,
) -> None:
    """Score experiment results against dataset labels."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    experiment_id = (
        pick_experiment_id(root)
        if experiment_id is None
        else resolve_experiment_arg(root, experiment_id, impl=implementation, dataset=dataset)
    )

    exp_dir = root / 'experiments' / experiment_id
    if not exp_dir.exists():
        console.print(f'[red]Experiment not found: {experiment_id}[/red]')
        raise typer.Exit(1)

    report = score_experiment(root, experiment_id)
    if get_output_mode().use_rich:
        print_eval_report(report)
    else:
        print(json.dumps(asdict(report), indent=2))

    if save:
        eval_path = exp_dir / 'eval.json'
        eval_path.write_text(json.dumps(asdict(report), indent=2))

        append_to_index(root, report)
        if get_output_mode().use_rich:
            metadata, _ = load_results(exp_dir)
            console.print(f'[green]Saved eval report for {format_ref_medium(metadata)}[/green]')
