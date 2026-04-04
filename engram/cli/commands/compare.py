"""Compare command: compare two experiments."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from engram.config.discovery import find_project_root
from engram.tracking.comparison import compare_experiments, diff_config_snapshots

console = Console()


def compare_command(
    experiment_a: Annotated[str, typer.Argument(help='First experiment ID')],
    experiment_b: Annotated[str, typer.Argument(help='Second experiment ID')],
    prompts: Annotated[bool, typer.Option('--prompts', help='Show full prompt diffs')] = False,
) -> None:
    """Compare two experiments: accuracy deltas, cost, config diffs."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    result = compare_experiments(root, experiment_a, experiment_b)

    # Accuracy table
    table = Table(title='Accuracy Comparison')
    table.add_column('Field', style='bold')
    table.add_column(experiment_a, justify='right')
    table.add_column(experiment_b, justify='right')
    table.add_column('Delta', justify='right')

    for delta in result.field_deltas.values():
        color = 'red' if delta.regressed else 'green'
        sign = '+' if delta.delta >= 0 else ''
        table.add_row(
            delta.field_name,
            f'{delta.accuracy_a:.1%}',
            f'{delta.accuracy_b:.1%}',
            f'[{color}]{sign}{delta.delta:.1%}[/{color}]',
        )

    console.print(table)
    console.print()

    # Cost table
    cost_table = Table(title='Cost Comparison')
    cost_table.add_column('Metric', style='bold')
    cost_table.add_column(experiment_a, justify='right')
    cost_table.add_column(experiment_b, justify='right')

    cost_table.add_row('Total', f'${result.cost_a.get("total", 0):.4f}', f'${result.cost_b.get("total", 0):.4f}')
    cost_table.add_row('Average', f'${result.cost_a.get("avg", 0):.4f}', f'${result.cost_b.get("avg", 0):.4f}')

    console.print(cost_table)
    console.print()

    # Config diff
    diff_lines = diff_config_snapshots(root, experiment_a, experiment_b, show_prompts=prompts)
    if diff_lines:
        console.print('[bold]Config Changes:[/bold]')
        for line in diff_lines:
            console.print(f'  {line}')
        console.print()

    # Regressions
    if result.regressions:
        console.print(f'[red bold]Regressions detected:[/red bold] {", ".join(result.regressions)}')
