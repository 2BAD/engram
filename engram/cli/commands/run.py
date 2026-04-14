"""Run command: execute a workflow against a dataset."""

import os
from typing import Annotated

import typer
from rich.console import Console

from engram.cli.completions import complete_datasets, complete_implementations
from engram.cli.prompts import ask_dataset, ask_implementation, ask_label, is_interactive
from engram.config.discovery import find_project_root
from engram.config.loader import load_implementation
from engram.eval.loop import run_eval
from engram.runners.registry import get_runner
from engram.tracking.index import compute_short_ids

console = Console()


def run_command(  # noqa: PLR0913 — CLI options map 1:1 to flags
    implementation: Annotated[
        str | None,
        typer.Argument(
            help='Implementation name. Omit to pick interactively.', autocompletion=complete_implementations
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            '--dataset', '-d', help='Dataset name. Omit to pick interactively.', autocompletion=complete_datasets
        ),
    ] = None,
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

    if is_interactive():
        implementation = implementation or ask_implementation(root)
        dataset = dataset or ask_dataset(root)
        if label is None:
            label = ask_label()
    elif not implementation or not dataset:
        console.print('[red]Implementation and --dataset are required in non-interactive mode.[/red]')
        raise typer.Exit(1)

    # Preflight: fail fast with a friendly message if any required env vars are
    # missing, instead of launching N worker threads that each raise KeyError.
    impl_config = load_implementation(root, implementation)
    runner = get_runner(impl_config.runner)
    missing = [v for v in runner.required_env_vars(impl_config) if v not in os.environ]
    if missing:
        console.print(f'[red]Missing required environment variable(s): {", ".join(missing)}[/red]')
        console.print('Add them to [bold].env[/bold] in the project root, or export them in your shell:')
        for var in missing:
            console.print(f'  [cyan]export {var}=...[/cyan]')
        raise typer.Exit(1)

    experiment_id = run_eval(
        root,
        implementation,
        dataset,
        concurrency,
        limit=limit,
        sample_seed=sample_seed,
        repeats=repeats,
        label=label,
    )
    short_id = compute_short_ids(root).get(experiment_id)
    ref = f'#{short_id}' if short_id is not None else experiment_id
    label_suffix = f' \\[{label.strip()}]' if label else ''
    console.print(f'[green]Experiment complete:[/green] {ref} [dim]{implementation}/{dataset}{label_suffix}[/dim]')
    console.print()
    console.print('[bold]Next steps:[/bold]')
    console.print(f'  Score the run:   [cyan]engram score {ref} --save[/cyan]')
    console.print('  List past runs:  [cyan]engram experiments list[/cyan]')
