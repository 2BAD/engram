"""Score command: score experiment results against labels."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console

from engram.config.discovery import find_project_root
from engram.display.tables import print_eval_report
from engram.scoring.engine import score_experiment
from engram.tracking.index import append_to_index

console = Console()


def score_command(
    experiment_id: Annotated[str, typer.Argument(help='Experiment ID to score')],
    save: Annotated[bool, typer.Option('--save', help='Save report and update experiment index')] = False,
) -> None:
    """Score experiment results against dataset labels."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    exp_dir = root / 'experiments' / experiment_id
    if not exp_dir.exists():
        console.print(f'[red]Experiment not found: {experiment_id}[/red]')
        raise typer.Exit(1)

    report = score_experiment(root, experiment_id)
    print_eval_report(report)

    if save:
        report_path = exp_dir / 'report.json'
        report_path.write_text(json.dumps(asdict(report), indent=2))

        append_to_index(root, report)
        console.print(f'[green]Report saved to {report_path}[/green]')
