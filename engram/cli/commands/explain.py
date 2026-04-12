"""Explain command: LLM-powered analysis of experiment results."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

from engram.analysis.analyzer import (
    AnalysisResult,
    call_llm,
    estimate_analysis_cost,
    load_cached,
    save_cached,
)
from engram.analysis.context import build_comparison_context, build_single_context
from engram.analysis.prompts import EXPLAIN_SYSTEM_PROMPT, build_comparison_message, build_single_message
from engram.cli.picker import pick_experiment_id, resolve_experiment_arg
from engram.cli.prompts import ask_confirm, is_interactive
from engram.config.discovery import find_project_root
from engram.config.loader import load_project
from engram.observability.output_mode import get_output_mode

console = Console()


def explain_command(
    experiment_a: Annotated[
        str | None,
        typer.Argument(help='Experiment ID, #N, or @ / @~N. Omit to pick interactively.'),
    ] = None,
    experiment_b: Annotated[
        str | None,
        typer.Argument(help='Optional second experiment for comparison analysis.'),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option('--yes', '-y', help='Skip cost confirmation prompt'),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option('--no-cache', help='Ignore cached analysis and re-run'),
    ] = False,
    implementation: Annotated[
        str | None,
        typer.Option('--impl', '-i', help='Scope @ / @~N resolution to this implementation'),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option('--dataset', '-d', help='Scope @ / @~N resolution to this dataset'),
    ] = None,
) -> None:
    """Explain experiment results using an LLM."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    project = load_project(root)
    if project.analysis is None:
        console.print('[red]No analysis config in engram.yaml.[/red]')
        console.print('Add an [bold]analysis:[/bold] block with at least a [bold]model:[/bold] field:')
        console.print()
        console.print('  analysis:')
        console.print('    model: claude-sonnet-4-5-20250514')
        raise typer.Exit(1)

    config = project.analysis

    experiment_a = (
        pick_experiment_id(root)
        if experiment_a is None
        else resolve_experiment_arg(root, experiment_a, impl=implementation, dataset=dataset)
    )
    if experiment_b is not None:
        experiment_b = resolve_experiment_arg(root, experiment_b, impl=implementation, dataset=dataset)

    is_comparison = experiment_b is not None

    # Check cache for single-experiment mode
    if not no_cache and not is_comparison:
        exp_dir = root / 'experiments' / experiment_a
        cached = load_cached(exp_dir)
        if cached is not None:
            _display_analysis(cached, experiment_a=experiment_a)
            return

    # Build context and estimate cost
    if experiment_b is not None:
        context = build_comparison_context(root, experiment_a, experiment_b, config.max_examples)
        user_message = build_comparison_message(context)
    else:
        context = build_single_context(root, experiment_a, config.max_examples)
        user_message = build_single_message(context)

    estimated_cost, est_input, est_output = estimate_analysis_cost(config, EXPLAIN_SYSTEM_PROMPT, user_message)

    if not _confirm_cost(config.model, estimated_cost, est_input, est_output, yes):
        raise typer.Exit(0)

    # Run analysis
    result = call_llm(config, EXPLAIN_SYSTEM_PROMPT, user_message)

    # Cache single-experiment results
    if not is_comparison:
        exp_dir = root / 'experiments' / experiment_a
        save_cached(exp_dir, result)

    _display_analysis(result.markdown, experiment_a=experiment_a, experiment_b=experiment_b, result=result)


def _confirm_cost(
    model: str,
    estimated_cost: float,
    est_input: int,
    est_output: int,
    skip: bool,
) -> bool:
    """Show estimated cost and ask for confirmation. Returns True to proceed."""
    if get_output_mode().use_rich:
        console.print(f'[bold]Analysis model:[/bold] {model}', highlight=False)
        console.print(f'[bold]Estimated tokens:[/bold] ~{est_input:,} input + ~{est_output:,} output')
        console.print(f'[bold]Estimated cost:[/bold] [yellow]${estimated_cost:.4f}[/yellow]')
        console.print()

    if skip or not is_interactive():
        return True

    return ask_confirm('Proceed with analysis?', default=True)


def _display_analysis(
    markdown_text: str,
    experiment_a: str,
    experiment_b: str | None = None,
    result: AnalysisResult | None = None,
) -> None:
    """Render the analysis output."""
    if not get_output_mode().use_rich:
        payload: dict = {'analysis': markdown_text}
        if result is not None:
            payload['model'] = result.model
            payload['input_tokens'] = result.input_tokens
            payload['output_tokens'] = result.output_tokens
            payload['cost_usd'] = result.cost_usd
        print(json.dumps(payload, indent=2))
        return

    if experiment_b:
        console.print(f'[bold]Analysis: {experiment_a} vs {experiment_b}[/bold]')
    else:
        console.print(f'[bold]Analysis: {experiment_a}[/bold]')
    console.print()
    console.print(Markdown(markdown_text))

    if result is not None:
        console.print()
        console.print(
            f'[dim]Model: {result.model} | '
            f'Tokens: {result.input_tokens:,} in + {result.output_tokens:,} out | '
            f'Cost: ${result.cost_usd:.4f}[/dim]'
        )
