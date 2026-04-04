"""Engram CLI."""

from typing import Annotated

import typer

from engram.observability.logging import configure_logging
from engram.observability.output_mode import OutputMode, set_output_mode

from .commands import compare_command, estimate_command, init_command, run_command, score_command, status_command

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


app.command(name='compare')(compare_command)
app.command(name='estimate')(estimate_command)
app.command(name='init')(init_command)
app.command(name='run')(run_command)
app.command(name='score')(score_command)
app.command(name='status')(status_command)
