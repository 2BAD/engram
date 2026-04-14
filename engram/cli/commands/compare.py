"""Compare command: compare two experiments."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from engram.cli.completions import complete_datasets, complete_experiment_ids, complete_implementations
from engram.cli.picker import pick_experiment_pair, resolve_experiment_arg
from engram.config.discovery import find_project_root
from engram.display.experiment_ref import format_ref_medium, format_ref_short, linkify_ref
from engram.eval.results import load_results
from engram.observability.output_mode import get_output_mode
from engram.tracking.baseline import get_workflow_baseline, lookup_experiment
from engram.tracking.comparison import ComparisonResult, compare_experiments, diff_config_snapshots

console = Console()

_NA_CELL = '[dim]—[/dim]'


_METRICS = ('accuracy', 'precision', 'recall', 'f1')


def _print_field_table(delta, from_ref: str, to_ref: str) -> None:
    """Render one table per field with all metrics as rows."""
    table = Table(title=delta.field_name)
    table.add_column(Text('Metric', justify='center'), style='bold')
    table.add_column(Text(from_ref, justify='center'), justify='right')
    table.add_column(Text(to_ref, justify='center'), justify='right')
    table.add_column(Text('Delta', justify='center'), justify='right')

    for metric in _METRICS:
        if metric != 'accuracy' and not delta.is_classification:
            table.add_row(metric.capitalize(), _NA_CELL, _NA_CELL, _NA_CELL)
            continue

        a_val = getattr(delta, f'{metric}_a')
        b_val = getattr(delta, f'{metric}_b')
        delta_val = b_val - a_val
        color = 'red' if delta_val < 0 else 'green'
        sign = '+' if delta_val >= 0 else ''
        table.add_row(
            metric.capitalize(),
            f'{a_val:.1%}',
            f'{b_val:.1%}',
            f'[{color}]{sign}{delta_val:.1%}[/{color}]',
        )

    console.print(table)
    console.print()


def _short_header(meta: dict) -> str:
    """``#N label`` if a label exists, otherwise just ``#N``."""
    short = format_ref_short(meta)
    label = meta.get('label')
    return f'{short} [{label}]' if label else short


def _warn_cross_workflow(root: Path, from_id: str, to_id: str) -> None:
    """Print a warning if the two experiments belong to different workflows."""
    try:
        wf_a, _ = lookup_experiment(root, from_id)
        wf_b, _ = lookup_experiment(root, to_id)
    except (FileNotFoundError, OSError, KeyError):
        return
    if wf_a != wf_b:
        console.print(
            f'[yellow]Warning: comparing across workflows ({wf_a} vs {wf_b}). Output schemas may differ.[/yellow]'
        )
        console.print()


def _warn_labels_drift(result: ComparisonResult, from_meta: dict, to_meta: dict) -> None:
    """Warn when two same-dataset experiments were scored against different label payloads."""
    if from_meta.get('dataset') != to_meta.get('dataset'):
        return
    hash_a = result.labels_a.get('hash')
    hash_b = result.labels_b.get('hash')
    if not hash_a or not hash_b or hash_a == hash_b:
        return
    console.print(
        '[yellow]Warning: label set differs between these experiments — '
        'accuracy deltas mix model change with ground-truth change.[/yellow]'
    )
    console.print(
        f'  A: hash {str(hash_a)[:12]}  count {result.labels_a.get("count")}  scored {result.labels_a.get("scored_at")}'
    )
    console.print(
        f'  B: hash {str(hash_b)[:12]}  count {result.labels_b.get("count")}  scored {result.labels_b.get("scored_at")}'
    )
    console.print("  Re-score the older experiment to compare against today's labels.")
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
        typer.Argument(
            help='Experiment ID, #N, or @ / @~N. Omit to pick interactively from recent runs.',
            autocompletion=complete_experiment_ids,
        ),
    ] = None,
    experiment_b: Annotated[
        str | None,
        typer.Argument(
            help='Optional second experiment ID. Defaults to the workflow baseline.',
            autocompletion=complete_experiment_ids,
        ),
    ] = None,
    against: Annotated[
        str | None,
        typer.Option(
            '--against',
            help='Compare against this specific experiment instead of the workflow baseline',
            autocompletion=complete_experiment_ids,
        ),
    ] = None,
    prompts: Annotated[bool, typer.Option('--prompts', help='Show full prompt diffs')] = False,
    implementation: Annotated[
        str | None,
        typer.Option(
            '--impl',
            '-i',
            help='Scope @ / @~N resolution to this implementation',
            autocompletion=complete_implementations,
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            '--dataset', '-d', help='Scope @ / @~N resolution to this dataset', autocompletion=complete_datasets
        ),
    ] = None,
) -> None:
    """Compare two experiments: accuracy deltas, cost, config diffs."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    if experiment_a is None:
        from_id, to_id = pick_experiment_pair(root)
    else:
        experiment_a = resolve_experiment_arg(root, experiment_a, impl=implementation, dataset=dataset)
        if experiment_b is not None:
            experiment_b = resolve_experiment_arg(root, experiment_b, impl=implementation, dataset=dataset)
        if against is not None:
            against = resolve_experiment_arg(root, against, impl=implementation, dataset=dataset)
        from_id, to_id = _resolve_compare_pair(root, experiment_a, experiment_b, against)

    _warn_cross_workflow(root, from_id, to_id)

    result = compare_experiments(root, from_id, to_id)
    diff_lines = diff_config_snapshots(root, from_id, to_id, show_prompts=prompts)

    if not get_output_mode().use_rich:
        payload = asdict(result)
        payload['config_changes'] = diff_lines
        print(json.dumps(payload, indent=2))
        return

    from_meta, _ = load_results(root / 'experiments' / from_id)
    to_meta, _ = load_results(root / 'experiments' / to_id)
    from_short = _short_header(from_meta)
    to_short = _short_header(to_meta)

    console.print(f'  {linkify_ref(format_ref_medium(from_meta), root / "experiments" / from_id)}')
    console.print(f'  {linkify_ref(format_ref_medium(to_meta), root / "experiments" / to_id)}')
    console.print()

    _warn_labels_drift(result, from_meta, to_meta)

    for delta in result.field_deltas.values():
        _print_field_table(delta, from_short, to_short)

    # Cost table
    cost_table = Table(title='Cost Comparison')
    cost_table.add_column(Text('Metric', justify='center'), style='bold')
    cost_table.add_column(Text(from_short, justify='center'), justify='right')
    cost_table.add_column(Text(to_short, justify='center'), justify='right')

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
