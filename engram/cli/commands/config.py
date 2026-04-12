"""Config command: manage hosted platform configs (pull/push/diff/deploy/list)."""

from typing import Annotated

import typer
from rich.console import Console

from engram.cli.completions import complete_implementations
from engram.config.discovery import discover_implementations, find_project_root
from engram.config.loader import load_implementation
from engram.config.sync import deploy_config, diff_config, pull_config, push_config

console = Console()

config_app = typer.Typer(name='config', help='Manage hosted platform configs.', no_args_is_help=True)


@config_app.command()
def pull(
    implementation: Annotated[str, typer.Argument(help='Implementation name', autocompletion=complete_implementations)],
) -> None:
    """Pull config from hosted platform to local files."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    impl_config = load_implementation(root, implementation)
    impl_dir = root / 'implementations' / implementation

    manifest = pull_config(impl_dir, impl_config)
    models = sorted({n['model'] for n in manifest.get('nodes', [])})
    console.print(f'[green]Pulled config for {implementation}[/green]')
    console.print(f'  {len(manifest.get("nodes", []))} nodes, models: {", ".join(models)}')


@config_app.command()
def diff(
    implementation: Annotated[str, typer.Argument(help='Implementation name', autocompletion=complete_implementations)],
) -> None:
    """Compare local config vs remote platform."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    impl_config = load_implementation(root, implementation)
    impl_dir = root / 'implementations' / implementation

    lines = diff_config(impl_dir, impl_config)
    if lines:
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                console.print(f'[green]{line}[/green]')
            elif line.startswith('-') and not line.startswith('---'):
                console.print(f'[red]{line}[/red]')
            elif line.startswith('@@'):
                console.print(f'[cyan]{line}[/cyan]')
            else:
                console.print(line)
    else:
        console.print('[green]No changes. Local matches remote.[/green]')


@config_app.command()
def push(
    implementation: Annotated[str, typer.Argument(help='Implementation name', autocompletion=complete_implementations)],
    dry_run: Annotated[bool, typer.Option('--dry-run', help='Show changes without saving')] = False,
) -> None:
    """Push local config changes to hosted platform."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    impl_config = load_implementation(root, implementation)
    impl_dir = root / 'implementations' / implementation

    changes = push_config(impl_dir, impl_config, dry_run=dry_run)
    if changes:
        for change in changes:
            console.print(f'  {change}')
        if dry_run:
            console.print('[yellow]Dry run. No changes saved.[/yellow]')
        else:
            console.print('[green]Pushed.[/green]')
    else:
        console.print('[green]No changes to push.[/green]')


@config_app.command()
def deploy(
    implementation: Annotated[str, typer.Argument(help='Implementation name', autocompletion=complete_implementations)],
    dry_run: Annotated[bool, typer.Option('--dry-run', help='Show changes without deploying')] = False,
) -> None:
    """Push config changes and deploy to hosted platform."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    impl_config = load_implementation(root, implementation)
    impl_dir = root / 'implementations' / implementation

    changes = deploy_config(impl_dir, impl_config, dry_run=dry_run)
    for change in changes:
        console.print(f'  {change}')
    if dry_run:
        console.print('[yellow]Dry run. No changes deployed.[/yellow]')


@config_app.command(name='list')
def list_configs() -> None:
    """List all implementations and their config status."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    implementations = discover_implementations(root)
    if not implementations:
        console.print('[dim]No implementations found.[/dim]')
        return

    for name in implementations:
        impl_config = load_implementation(root, name)
        mode = impl_config.config_management.mode
        manifest_path = root / 'implementations' / name / 'manifest.json'
        has_manifest = manifest_path.exists()
        status = '[green]synced[/green]' if has_manifest else '[dim]local only[/dim]'
        if mode == 'pull-push':
            console.print(f'  {name} ({impl_config.platform}/{impl_config.runner}) [{mode}] {status}')
        else:
            console.print(f'  {name} ({impl_config.platform}/{impl_config.runner}) [{mode}]')
