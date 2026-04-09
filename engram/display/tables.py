"""Rich table formatters for scoring results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from engram.models.scoring import ConfusionMatrix, EvalReport

console = Console()


def print_eval_report(report: EvalReport) -> None:
    """Print a formatted evaluation report."""
    _print_metrics_table(report)
    _print_cost_table(report)

    for cm in report.confusion_matrices:
        _print_confusion_matrix(cm)


_THRESHOLD_GREEN = 0.8
_THRESHOLD_YELLOW = 0.6
_NA_CELL = '[dim]—[/dim]'


def _score_color(value: float) -> str:
    if value >= _THRESHOLD_GREEN:
        return 'green'
    if value >= _THRESHOLD_YELLOW:
        return 'yellow'
    return 'red'


def _format_score(value: float) -> str:
    color = _score_color(value)
    return f'[{color}]{value:.1%}[/{color}]'


def _print_metrics_table(report: EvalReport) -> None:
    table = Table(title='Field Metrics')
    table.add_column('Field', style='bold')
    table.add_column('Accuracy', justify='right')
    table.add_column('Precision', justify='right')
    table.add_column('Recall', justify='right')
    table.add_column('F1', justify='right')
    table.add_column('N', justify='right')

    for fm in report.field_metrics:
        if fm.is_classification:
            row = [
                fm.field_name,
                _format_score(fm.accuracy),
                _format_score(fm.precision),
                _format_score(fm.recall),
                _format_score(fm.f1),
                str(fm.total),
            ]
        else:
            row = [
                fm.field_name,
                _format_score(fm.accuracy),
                _NA_CELL,
                _NA_CELL,
                _NA_CELL,
                str(fm.total),
            ]
        table.add_row(*row)

    console.print(table)
    console.print()


def _print_cost_table(report: EvalReport) -> None:
    if report.cost_total_usd == 0:
        return

    table = Table(title='Cost')
    table.add_column('Metric', style='bold')
    table.add_column('Value', justify='right')

    table.add_row('Total', f'${report.cost_total_usd:.4f}')
    table.add_row('Average', f'${report.cost_avg_usd:.4f}')
    table.add_row('Median', f'${report.cost_median_usd:.4f}')
    table.add_row('P95', f'${report.cost_p95_usd:.4f}')

    console.print(table)
    console.print()


def _print_confusion_matrix(cm: ConfusionMatrix) -> None:
    table = Table(title=f'Confusion Matrix: {cm.field_name}')
    table.add_column('Expected \\ Predicted', style='bold')
    for label in cm.labels:
        table.add_column(label, justify='right')

    for expected in cm.labels:
        row = [expected]
        for predicted in cm.labels:
            count = cm.matrix.get(expected, {}).get(predicted, 0)
            if expected == predicted:
                row.append(f'[bold]{count}[/bold]')
            elif count > 0:
                row.append(f'[red]{count}[/red]')
            else:
                row.append(str(count))
        table.add_row(*row)

    console.print(table)
    console.print()
