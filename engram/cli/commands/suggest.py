"""Suggest command: LLM-powered recommendations for improving experiment results."""

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
from engram.analysis.context import build_single_context
from engram.analysis.prompts import SUGGEST_SYSTEM_PROMPT, build_single_message
from engram.cli.picker import pick_experiment_id, resolve_experiment_arg
from engram.cli.prompts import ask_confirm, is_interactive
from engram.config.discovery import find_project_root
from engram.config.loader import load_project
from engram.observability.output_mode import get_output_mode

console = Console()

_CACHE_FILENAME = 'suggest.md'


def suggest_command(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help='Experiment ID, short_id, or @ / @~N. Omit to pick interactively.'),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option('--yes', '-y', help='Skip cost confirmation prompt'),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option('--no-cache', help='Ignore cached suggestions and re-run'),
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
    """Suggest improvements for experiment results using an LLM."""
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

    experiment_id = (
        pick_experiment_id(root)
        if experiment_id is None
        else resolve_experiment_arg(root, experiment_id, impl=implementation, dataset=dataset)
    )

    exp_dir = root / 'experiments' / experiment_id

    # Check cache
    if not no_cache:
        cached = load_cached(exp_dir, _CACHE_FILENAME)
        if cached is not None:
            _display_suggestions(cached, experiment_id)
            return

    # Build context and estimate cost
    context = build_single_context(root, experiment_id, config.max_examples)
    user_message = build_single_message(context)

    estimated_cost, est_input, est_output = estimate_analysis_cost(config, SUGGEST_SYSTEM_PROMPT, user_message)

    if not _confirm_cost(config.model, estimated_cost, est_input, est_output, yes):
        raise typer.Exit(0)

    result = call_llm(config, SUGGEST_SYSTEM_PROMPT, user_message)

    save_cached(exp_dir, result, _CACHE_FILENAME)

    _display_suggestions(result.markdown, experiment_id, result=result)


def _confirm_cost(
    model: str,
    estimated_cost: float,
    est_input: int,
    est_output: int,
    skip: bool,
) -> bool:
    """Show estimated cost and ask for confirmation. Returns True to proceed."""
    if get_output_mode().use_rich:
        console.print(f'[bold]Analysis model:[/bold] {model}')
        console.print(f'[bold]Estimated tokens:[/bold] ~{est_input:,} input + ~{est_output:,} output')
        console.print(f'[bold]Estimated cost:[/bold] [yellow]${estimated_cost:.4f}[/yellow]')
        console.print()

    if skip or not is_interactive():
        return True

    return ask_confirm('Proceed with analysis?', default=True)


def _display_suggestions(
    markdown_text: str,
    experiment_id: str,
    result: AnalysisResult | None = None,
) -> None:
    """Render the suggestions output."""
    if not get_output_mode().use_rich:
        payload: dict = {'suggestions': markdown_text}
        if result is not None:
            payload['model'] = result.model
            payload['input_tokens'] = result.input_tokens
            payload['output_tokens'] = result.output_tokens
            payload['cost_usd'] = result.cost_usd
        print(json.dumps(payload, indent=2))
        return

    console.print(f'[bold]Suggestions: {experiment_id}[/bold]')
    console.print()
    console.print(Markdown(markdown_text))

    if result is not None:
        console.print()
        console.print(
            f'[dim]Model: {result.model} | '
            f'Tokens: {result.input_tokens:,} in + {result.output_tokens:,} out | '
            f'Cost: ${result.cost_usd:.4f}[/dim]'
        )
