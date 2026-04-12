"""Run command: execute a workflow against a dataset."""

import os
from typing import Annotated

import typer
from rich.console import Console

from engram.config.discovery import find_project_root
from engram.config.loader import load_implementation
from engram.eval.loop import run_eval
from engram.runners.registry import get_runner

console = Console()


def run_command(  # noqa: PLR0913 — CLI options map 1:1 to flags
    implementation: Annotated[str, typer.Argument(help='Implementation name')],
    dataset: Annotated[str, typer.Option('--dataset', '-d', help='Dataset name')],
    concurrency: Annotated[int, typer.Option('--concurrency', '-c', help='Number of concurrent runs')] = 5,
    limit: Annotated[
        int | None,
        typer.Option('--limit', '-n', help='Sample N inputs from the dataset (deterministic with --sample-seed)'),
    ] = None,
    sample_seed: Annotated[
        int,
        typer.Option(
            '--sample-seed',
            help='RNG seed for dataset subsampling only; does not seed model sampling',
        ),
    ] = 0,
    repeats: Annotated[
        int,
        typer.Option(
            '--repeat',
            '-r',
            help='Run each input N times to measure run-to-run noise. '
            'Total triggers = inputs * repeats; concurrency is not scaled.',
        ),
    ] = 1,
    label: Annotated[
        str | None,
        typer.Option(
            '--label',
            '-l',
            help='Optional human-readable label for this run (e.g. "prompt-v2", "before-refactor").',
        ),
    ] = None,
) -> None:
    """Evaluate a workflow implementation against a dataset."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    # Preflight: fail fast with a friendly message if any required env vars are
    # missing, instead of launching N worker threads that each raise KeyError.
    impl_config = load_implementation(root, implementation)
    runner = get_runner(impl_config.runner)
    missing = [v for v in runner.required_env_vars(impl_config) if v not in os.environ]
    if missing:
        console.print(
            f'[red]Missing required environment variable(s): {", ".join(missing)}[/red]'
        )
        console.print(
            'Add them to [bold].env[/bold] in the project root, or export them in your shell:'
        )
        for var in missing:
            console.print(f'  [cyan]export {var}=...[/cyan]')
        raise typer.Exit(1)

    _experiment_id, short_id = run_eval(
        root,
        implementation,
        dataset,
        concurrency,
        limit=limit,
        sample_seed=sample_seed,
        repeats=repeats,
        label=label,
    )
    label_suffix = f' \\[{label.strip()}]' if label else ''
    console.print(
        f'[green]Experiment complete:[/green] #{short_id} [dim]{implementation}/{dataset}{label_suffix}[/dim]'
    )
    console.print()
    console.print('[bold]Next steps:[/bold]')
    console.print(f'  Score the run:   [cyan]engram score {short_id} --save[/cyan]')
    console.print('  List past runs:  [cyan]engram experiments list[/cyan]')
