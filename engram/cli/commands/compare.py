"""Compare command: compare two experiments."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from engram.cli.picker import pick_experiment_id, resolve_experiment_arg
from engram.config.discovery import find_project_root
from engram.display.experiment_ref import format_ref_medium
from engram.eval.results import load_results
from engram.observability.output_mode import get_output_mode
from engram.tracking.baseline import get_workflow_baseline, lookup_experiment
from engram.tracking.comparison import ComparisonResult, compare_experiments, diff_config_snapshots

console = Console()

_NA_CELL = '[dim]—[/dim]'


def _print_metric_table(title: str, result: ComparisonResult, from_ref: str, to_ref: str, metric: str) -> None:
    """Render one (Field, A, B, Δ) table for a single metric across all field deltas."""
    table = Table(title=title)
    table.add_column('Field', style='bold')
    table.add_column(from_ref, justify='right')
    table.add_column(to_ref, justify='right')
    table.add_column('Delta', justify='right')

    for delta in result.field_deltas.values():
        # Accuracy is always meaningful; the other three require classification.
        if metric != 'accuracy' and not delta.is_classification:
            table.add_row(delta.field_name, _NA_CELL, _NA_CELL, _NA_CELL)
            continue

        a_val = getattr(delta, f'{metric}_a')
        b_val = getattr(delta, f'{metric}_b')
        delta_val = b_val - a_val
        color = 'red' if delta_val < 0 else 'green'
        sign = '+' if delta_val >= 0 else ''
        table.add_row(
            delta.field_name,
            f'{a_val:.1%}',
            f'{b_val:.1%}',
            f'[{color}]{sign}{delta_val:.1%}[/{color}]',
        )

    console.print(table)
    console.print()


def _resolve_compare_pair(
    root: Path,
    experiment_a: str,
    experiment_b: str | None,
    against: str | None,
) -> tuple[str, str]:
    """Derive the (from, to) pair for compare_experiments; baseline fallback kicks in when only `experiment_a` is given."""  # noqa: E501
    if experiment_b is not None:
        return experiment_a, experiment_b
    if against is not None:
        return against, experiment_a

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
    return baseline, experiment_a


def compare_command(
    experiment_a: Annotated[
        str | None,
        typer.Argument(help='Experiment ID, short_id, or @ / @~N. Omit to pick interactively from recent runs.'),
    ] = None,
    experiment_b: Annotated[
        str | None,
        typer.Argument(help='Optional second experiment ID. Defaults to the workflow baseline.'),
    ] = None,
    against: Annotated[
        str | None,
        typer.Option('--against', help='Compare against this specific experiment instead of the workflow baseline'),
    ] = None,
    prompts: Annotated[bool, typer.Option('--prompts', help='Show full prompt diffs')] = False,
    implementation: Annotated[
        str | None,
        typer.Option('--impl', '-i', help='Scope @ / @~N resolution to this implementation'),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option('--dataset', '-d', help='Scope @ / @~N resolution to this dataset'),
    ] = None,
) -> None:
    """Compare two experiments: accuracy deltas, cost, config diffs."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    experiment_a = (
        pick_experiment_id(root)
        if experiment_a is None
        else resolve_experiment_arg(root, experiment_a, impl=implementation, dataset=dataset)
    )
    if experiment_b is not None:
        experiment_b = resolve_experiment_arg(root, experiment_b, impl=implementation, dataset=dataset)
    if against is not None:
        against = resolve_experiment_arg(root, against, impl=implementation, dataset=dataset)

    from_id, to_id = _resolve_compare_pair(root, experiment_a, experiment_b, against)

    result = compare_experiments(root, from_id, to_id)
    diff_lines = diff_config_snapshots(root, from_id, to_id, show_prompts=prompts)

    if not get_output_mode().use_rich:
        payload = asdict(result)
        payload['config_changes'] = diff_lines
        print(json.dumps(payload, indent=2))
        return

    # Build pretty '#N impl/dataset' refs from each experiment's own metadata so
    # table headers never show the long underscore-joined full id.
    from_meta, _ = load_results(root / 'experiments' / from_id)
    to_meta, _ = load_results(root / 'experiments' / to_id)
    from_ref = escape(format_ref_medium(from_meta))
    to_ref = escape(format_ref_medium(to_meta))

    # Four stacked per-metric tables. Accuracy always shows real numbers; the other
    # three render "—" for non-classification fields (where they fall back to accuracy
    # and would be misleading to display as separate values).
    _print_metric_table('Accuracy Comparison', result, from_ref, to_ref, metric='accuracy')
    _print_metric_table('Precision Comparison', result, from_ref, to_ref, metric='precision')
    _print_metric_table('Recall Comparison', result, from_ref, to_ref, metric='recall')
    _print_metric_table('F1 Comparison', result, from_ref, to_ref, metric='f1')

    # Cost table
    cost_table = Table(title='Cost Comparison')
    cost_table.add_column('Metric', style='bold')
    cost_table.add_column(from_ref, justify='right')
    cost_table.add_column(to_ref, justify='right')

    cost_table.add_row('Total', f'${result.cost_a.get("total", 0):.4f}', f'${result.cost_b.get("total", 0):.4f}')
    cost_table.add_row('Average', f'${result.cost_a.get("avg", 0):.4f}', f'${result.cost_b.get("avg", 0):.4f}')

    console.print(cost_table)
    console.print()

    if diff_lines:
        console.print('[bold]Config Changes:[/bold]')
        for line in diff_lines:
            console.print(f'  {line}')
        console.print()

    if result.regressions:
        console.print(f'[red bold]Regressions detected:[/red bold] {", ".join(result.regressions)}')
