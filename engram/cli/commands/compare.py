"""Compare command: compare two experiments."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from engram.config.discovery import find_project_root
from engram.tracking.baseline import get_workflow_baseline, lookup_experiment
from engram.tracking.comparison import compare_experiments, diff_config_snapshots

console = Console()


def compare_command(
    experiment_a: Annotated[str, typer.Argument(help='Experiment ID to compare')],
    experiment_b: Annotated[
        str | None,
        typer.Argument(help='Optional second experiment ID. Defaults to the workflow baseline.'),
    ] = None,
    against: Annotated[
        str | None,
        typer.Option('--against', help='Compare against this specific experiment instead of the workflow baseline'),
    ] = None,
    prompts: Annotated[bool, typer.Option('--prompts', help='Show full prompt diffs')] = False,
) -> None:
    """Compare two experiments: accuracy deltas, cost, config diffs."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    # Resolve the (from, to) pair. Two-arg mode preserves the original (A, B) order
    # exactly. Single-arg modes treat the user-supplied ID as the new ('to') side and
    # the baseline / --against as the 'from' side, so positive deltas read as wins.
    if experiment_b is not None:
        from_id, to_id = experiment_a, experiment_b
    elif against is not None:
        from_id, to_id = against, experiment_a
    else:
        try:
            workflow, _impl = lookup_experiment(root, experiment_a)
        except FileNotFoundError as e:
            console.print(f'[red]{e}[/red]')
            raise typer.Exit(1) from None
        baseline = get_workflow_baseline(root, workflow)
        if baseline is None:
            console.print(
                f'[red]No baseline set for workflow [bold]{workflow}[/bold].[/red]\n'
                '  Set one with [bold]engram baseline set <experiment-id>[/bold] '
                'or pass a second experiment ID explicitly.'
            )
            raise typer.Exit(1)
        from_id, to_id = baseline, experiment_a

    result = compare_experiments(root, from_id, to_id)

    # Accuracy table
    table = Table(title='Accuracy Comparison')
    table.add_column('Field', style='bold')
    table.add_column(from_id, justify='right')
    table.add_column(to_id, justify='right')
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
    cost_table.add_column(from_id, justify='right')
    cost_table.add_column(to_id, justify='right')

    cost_table.add_row('Total', f'${result.cost_a.get("total", 0):.4f}', f'${result.cost_b.get("total", 0):.4f}')
    cost_table.add_row('Average', f'${result.cost_a.get("avg", 0):.4f}', f'${result.cost_b.get("avg", 0):.4f}')

    console.print(cost_table)
    console.print()

    # Config diff
    diff_lines = diff_config_snapshots(root, from_id, to_id, show_prompts=prompts)
    if diff_lines:
        console.print('[bold]Config Changes:[/bold]')
        for line in diff_lines:
            console.print(f'  {line}')
        console.print()

    # Regressions
    if result.regressions:
        console.print(f'[red bold]Regressions detected:[/red bold] {", ".join(result.regressions)}')
