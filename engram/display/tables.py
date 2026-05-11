"""Rich table formatters for scoring results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from engram.models.scoring import ConfusionMatrix, EvalReport

console = Console()


def print_eval_report(report: EvalReport) -> None:
    """Print a formatted evaluation report."""
    _print_metrics_table(report)
    _print_tokens_table(report)
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
    # Repeat-aware columns appear only when at least one field carries the new metrics.
    # Single-repeat experiments render the same six-column table they did before.
    show_repeat_columns = any(fm.accuracy_stdev is not None for fm in report.field_metrics)

    table = Table(title='Field Metrics')
    table.add_column(Text('Field', justify='center'), style='bold')
    table.add_column(Text('Accuracy', justify='center'), justify='right')
    table.add_column(Text('Precision', justify='center'), justify='right')
    table.add_column(Text('Recall', justify='center'), justify='right')
    table.add_column(Text('F1', justify='center'), justify='right')
    table.add_column(Text('N', justify='center'), justify='right')
    if show_repeat_columns:
        table.add_column(Text('Stdev', justify='center'), justify='right')
        table.add_column(Text('Agree', justify='center'), justify='right')
        table.add_column(Text('Maj', justify='center'), justify='right')
        table.add_column(Text('Kappa', justify='center'), justify='right')

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
        if show_repeat_columns:
            row.extend(
                [
                    _format_optional_pct(fm.accuracy_stdev),
                    _format_optional_pct(fm.mean_agreement_rate),
                    _format_optional_pct(fm.majority_rate),
                    _format_optional_kappa(fm.fleiss_kappa),
                ]
            )
        table.add_row(*row)

    console.print(table)
    console.print()


def _format_optional_pct(value: float | None) -> str:
    if value is None:
        return _NA_CELL
    return f'{value:.1%}'


def _format_optional_kappa(value: float | None) -> str:
    if value is None:
        return _NA_CELL
    return f'{value:.2f}'


def _print_tokens_table(report: EvalReport) -> None:
    """
    Tokens table: prompt/cache-read/cache-write/completion totals with per-call averages.

    Hidden entirely when no token data was recorded. The prompt-bucket sub-rows and the hit-rate
    row only render when prompt caching was actually exercised; for a no-cache run the table is
    just Prompt/Completion/Total so it stays useful but quiet.
    """
    if report.tokens_prompt == 0 and report.tokens_completion == 0:
        return

    n_calls = _successful_call_count(report)
    has_cache = report.tokens_cache_read > 0 or report.tokens_cache_creation > 0

    table = Table(title='Tokens')
    table.add_column(Text('Metric', justify='center'), style='bold')
    table.add_column(Text('Total', justify='center'), justify='right')
    table.add_column(Text('Avg / call', justify='center'), justify='right')

    table.add_row('Prompt', _tokens(report.tokens_prompt), _tokens_avg(report.tokens_prompt, n_calls))
    if has_cache:
        uncached = max(report.tokens_prompt - report.tokens_cache_read - report.tokens_cache_creation, 0)
        table.add_row('  ├ uncached', _tokens(uncached), _tokens_avg(uncached, n_calls))
        table.add_row(
            '  ├ cache read',
            _tokens(report.tokens_cache_read),
            _tokens_avg(report.tokens_cache_read, n_calls),
        )
        table.add_row(
            '  └ cache write',
            _tokens(report.tokens_cache_creation),
            _tokens_avg(report.tokens_cache_creation, n_calls),
        )
    table.add_row('Completion', _tokens(report.tokens_completion), _tokens_avg(report.tokens_completion, n_calls))
    table.add_row(
        'Total',
        _tokens(report.tokens_prompt + report.tokens_completion),
        _tokens_avg(report.tokens_prompt + report.tokens_completion, n_calls),
    )
    if report.cache_hit_rate is not None:
        # Cache hit rate is an experiment-level proportion, no per-call value to show.
        table.add_row('Hit rate', f'{report.cache_hit_rate:.1%}', '')

    console.print(table)
    console.print()


def _tokens(n: int) -> str:
    return f'{n:,}'


def _tokens_avg(total: int, calls: int) -> str:
    if calls <= 0:
        return ''
    return f'{round(total / calls):,}'


def _successful_call_count(report: EvalReport) -> int:
    """
    Recover the number of successful calls from the matched_examples and completion total.

    Falls back to ``matched_examples`` when there's no usage data. Used only as a divisor for
    per-call averages in the tokens table.
    """
    if report.matched_examples > 0:
        return report.matched_examples
    return 1


def _print_cost_table(report: EvalReport) -> None:
    if report.cost_total_usd == 0:
        return

    table = Table(title='Cost')
    table.add_column(Text('Metric', justify='center'), style='bold')
    table.add_column(Text('Value', justify='center'), justify='right')

    table.add_row('Total', f'${report.cost_total_usd:.4f}')
    table.add_row('Average', f'${report.cost_avg_usd:.4f}')
    table.add_row('Median', f'${report.cost_median_usd:.4f}')
    table.add_row('P95', f'${report.cost_p95_usd:.4f}')

    # Bucket rows only render when the runner actually populated them — older runs
    # (or non-API runners) carry zeros across all four and would produce noise rows.
    buckets = (
        ('  ├ input', report.cost_input_usd),
        ('  ├ cache creation', report.cost_cache_creation_usd),
        ('  ├ cache read', report.cost_cache_read_usd),
        ('  └ output', report.cost_output_usd),
    )
    if any(amount > 0 for _, amount in buckets):
        for label, amount in buckets:
            table.add_row(label, f'${amount:.4f}')

    if report.cache_hit_rate is not None:
        table.add_row('Cache hit rate', f'{report.cache_hit_rate:.1%}')

    # Savings line: how much caching is worth vs the full-input-rate counterfactual.
    # Negative on a cold run that paid the creation premium without enough reads to recoup.
    if report.cost_without_cache_total_usd > 0:
        saved = report.cost_without_cache_total_usd - report.cost_total_usd
        pct = saved / report.cost_without_cache_total_usd if report.cost_without_cache_total_usd else 0.0
        table.add_row('Without cache', f'${report.cost_without_cache_total_usd:.4f}')
        if saved >= 0:
            table.add_row('Saved', f'[green]${saved:.4f} ({pct:.1%})[/green]')
        else:
            table.add_row('Saved', f'[red]-${abs(saved):.4f} ({abs(pct):.1%} overhead)[/red]')

    console.print(table)
    console.print()


def _print_confusion_matrix(cm: ConfusionMatrix) -> None:
    table = Table(title=f'Confusion Matrix: {cm.field_name}')
    table.add_column(Text('Expected \\ Predicted', justify='center'), style='bold')
    for label in cm.labels:
        table.add_column(Text(label, justify='center'), justify='right')

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
