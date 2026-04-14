"""Estimate command: estimate cost before running."""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from engram.cli.completions import complete_datasets, complete_implementations
from engram.cli.prompts import ask_dataset, ask_implementation, is_interactive
from engram.config.discovery import find_project_root
from engram.cost.estimator import estimate_cost
from engram.observability.output_mode import get_output_mode

console = Console()


def estimate_command(
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
) -> None:
    """Estimate cost for running an implementation against a dataset."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    if is_interactive():
        implementation = implementation or ask_implementation(root)
        dataset = dataset or ask_dataset(root)
    elif not implementation or not dataset:
        console.print('[red]Implementation and --dataset are required in non-interactive mode.[/red]')
        raise typer.Exit(1)

    result = estimate_cost(root, implementation, dataset)

    if not get_output_mode().use_rich:
        print(json.dumps(result, indent=2))
        return

    console.print(f'[bold]Cost Estimate:[/bold] {result["implementation"]} / {result["dataset"]}')
    console.print(f'  Model: {result["model"]}')
    console.print(f'  Examples: {result["total_examples"]}')
    console.print(f'  Prompt template tokens: {result["prompt_template_tokens"]}')
    console.print(f'  Avg output tokens (est): {result["avg_output_tokens"]}')
    for warning in result.get('warnings', []):
        console.print(f'  [yellow]{warning}[/yellow]')
    console.print()

    table = Table(title='Estimated Cost')
    table.add_column(Text('Metric', justify='center'), style='bold')
    table.add_column(Text('Value', justify='center'), justify='right')

    table.add_row('Input rate', f'${result["input_rate_per_token"]:.8f}/token')
    table.add_row('Output rate', f'${result["output_rate_per_token"]:.8f}/token')
    table.add_row('Total estimated', f'[bold]${result["total_estimated_cost_usd"]:.4f}[/bold]')

    console.print(table)
