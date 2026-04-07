"""Init command: scaffold a new engram project."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def init_command() -> None:
    """Scaffold a new engram project in the current directory."""
    root = Path.cwd()
    config_path = root / 'engram.yaml'

    if config_path.exists():
        console.print('[red]engram.yaml already exists in this directory.[/red]')
        raise typer.Exit(1)

    dirs = ['workflows', 'implementations', 'datasets', 'experiments']
    for d in dirs:
        (root / d).mkdir(exist_ok=True)

    # Gitignore experiment results
    (root / 'experiments' / '.gitignore').write_text('*\n!.gitignore\n!experiments.jsonl\n!baselines.json\n')

    config_path.write_text('name: my-project\ndescription: An engram evaluation project\n')

    console.print('[green]Initialized engram project.[/green]')
    console.print(f'  Config: {config_path}')
    for d in dirs:
        console.print(f'  Created: {d}/')
