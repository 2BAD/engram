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

    _render(result)


def _render(result: dict) -> None:
    avg_output_tokens = result['avg_output_tokens']

    console.print(f'[bold]Cost Estimate:[/bold] {result["implementation"]} / {result["dataset"]}')
    console.print(f'  Model: {result["model"]}')
    console.print(f'  Examples: {result["total_examples"]}')
    console.print(f'  Prompt template tokens: {result["prompt_template_tokens"]}')
    if avg_output_tokens is not None:
        console.print(f'  Avg output tokens (est): {avg_output_tokens}')
    else:
        console.print('  Avg output tokens (est): [dim]unknown - no prior run for calibration[/dim]')
    for warning in result.get('warnings', []):
        console.print(f'  [yellow]{warning}[/yellow]')
    console.print()

    table = Table(title='Estimated Cost')
    table.add_column(Text('Metric', justify='center'), style='bold')
    table.add_column(Text('Value', justify='center'), justify='right')

    table.add_row('Input rate', f'${result["input_rate_per_token"]:.8f}/token')
    table.add_row('Output rate', f'${result["output_rate_per_token"]:.8f}/token')
    table.add_row('Input cost', f'${result["estimated_input_cost_usd"]:.4f}')
    _add_output_and_total_rows(table, result)
    if 'estimated_savings_usd' in result:
        _add_cache_rows(table, result)

    console.print(table)


def _add_output_and_total_rows(table: Table, result: dict) -> None:
    output_cost = result['estimated_output_cost_usd']
    total = result['total_estimated_cost_usd']
    if output_cost is not None and total is not None:
        table.add_row('Output cost', f'${output_cost:.4f}')
        table.add_row('Total estimated', f'[bold]${total:.4f}[/bold]')
    else:
        table.add_row('Output cost', '[dim]not estimated[/dim]')
        table.add_row('Total estimated', '[dim]input only - run once to calibrate output[/dim]')


def _add_cache_rows(table: Table, result: dict) -> None:
    saved = result['estimated_savings_usd']
    without_cache = result.get('estimated_cost_without_cache_usd')
    if without_cache is not None:
        table.add_row('Without cache', f'${without_cache:.4f}')
        pct = saved / without_cache if without_cache else 0.0
        if saved >= 0:
            table.add_row('Saved', f'[green]${saved:.4f} ({pct:.1%})[/green]')
        else:
            table.add_row('Saved', f'[red]-${abs(saved):.4f} ({abs(pct):.1%} overhead)[/red]')
    elif saved >= 0:
        table.add_row('Saved (input side)', f'[green]${saved:.4f}[/green]')
    else:
        table.add_row('Saved (input side)', f'[red]-${abs(saved):.4f} overhead[/red]')
    if 'cache_hit_rate_used' in result:
        table.add_row('Hit rate (calibrated)', f'{result["cache_hit_rate_used"]:.1%}')
