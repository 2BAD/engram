"""Estimate command: estimate cost before running."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from engram.config.discovery import find_project_root
from engram.cost.estimator import estimate_cost

console = Console()


def estimate_command(
    implementation: Annotated[str, typer.Argument(help='Implementation name')],
    dataset: Annotated[str, typer.Option('--dataset', '-d', help='Dataset name')],
) -> None:
    """Estimate cost for running an implementation against a dataset."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    result = estimate_cost(root, implementation, dataset)

    console.print(f'[bold]Cost Estimate:[/bold] {result["implementation"]} / {result["dataset"]}')
    console.print(f'  Model: {result["model"]}')
    console.print(f'  Examples: {result["total_examples"]}')
    console.print(f'  Prompt template tokens: {result["prompt_template_tokens"]}')
    console.print(f'  Avg output tokens (est): {result["avg_output_tokens"]}')
    console.print()

    table = Table(title='Estimated Cost')
    table.add_column('Metric', style='bold')
    table.add_column('Value', justify='right')

    table.add_row('Input rate', f'${result["input_rate_per_token"]:.8f}/token')
    table.add_row('Output rate', f'${result["output_rate_per_token"]:.8f}/token')
    table.add_row('Total estimated', f'[bold]${result["total_estimated_cost_usd"]:.4f}[/bold]')

    console.print(table)
