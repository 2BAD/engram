"""Engram CLI."""

from typing import Annotated

import typer

from engram.config.discovery import find_project_root
from engram.config.env import load_project_env
from engram.observability.logging import configure_logging
from engram.observability.output_mode import OutputMode, set_output_mode

from .commands import (
    baseline_app,
    compare_command,
    config_app,
    estimate_command,
    experiments_app,
    explain_command,
    init_command,
    run_command,
    score_command,
    status_command,
    suggest_command,
    traces_app,
)
from .errors import run_with_error_handling

app = typer.Typer(
    name='engram',
    help='AI workflow evaluation and experimentation framework.',
    no_args_is_help=True,
)


@app.callback()
def main(
    json_output: Annotated[
        bool,
        typer.Option(
            '--json',
            help='Output JSON logs instead of Rich formatting (auto-detected for non-TTY)',
        ),
    ] = False,
) -> None:
    """Engram CLI."""
    mode = OutputMode.detect(force_json=json_output)
    set_output_mode(mode)
    configure_logging(json_format=mode.use_json_logging)

    # Populate os.environ from <project-root>/.env before any command runs, so API
    # keys and other runner config can live in the project dir instead of the shell.
    # Silently no-ops outside a project (e.g. `engram init`, `engram --help`).
    project_root = find_project_root()
    if project_root is not None:
        load_project_env(project_root)


def run() -> None:
    """CLI entry point with friendly error handling."""
    run_with_error_handling(app)


app.add_typer(baseline_app)
app.command(name='compare')(compare_command)
app.add_typer(config_app)
app.command(name='estimate')(estimate_command)
app.command(name='explain')(explain_command)
app.command(name='run')(run_command)
app.add_typer(experiments_app)
app.command(name='init')(init_command)
app.command(name='score')(score_command)
app.command(name='status')(status_command)
app.command(name='suggest')(suggest_command)
app.add_typer(traces_app)
