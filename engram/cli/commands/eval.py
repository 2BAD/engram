"""Eval command: execute a workflow against a dataset."""

from typing import Annotated

import typer
from rich.console import Console

from engram.config.discovery import find_project_root
from engram.eval.loop import run_eval

console = Console()


def eval_command(
    implementation: Annotated[str, typer.Argument(help='Implementation name')],
    dataset: Annotated[str, typer.Option('--dataset', '-d', help='Dataset name')],
    concurrency: Annotated[int, typer.Option('--concurrency', '-c', help='Number of concurrent runs')] = 5,
    limit: Annotated[
        int | None,
        typer.Option('--limit', '-n', help='Sample N inputs from the dataset (deterministic with --seed)'),
    ] = None,
    seed: Annotated[int, typer.Option('--seed', help='RNG seed for sampling; same seed produces the same subset')] = 0,
) -> None:
    """Evaluate a workflow implementation against a dataset."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    experiment_id = run_eval(root, implementation, dataset, concurrency, limit=limit, seed=seed)
    console.print(f'[green]Experiment complete:[/green] {experiment_id}')
