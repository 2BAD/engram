"""Traces command: pull and cache traces from hosted platforms."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated

import typer
from rich.console import Console

from engram.config.discovery import find_project_root
from engram.config.loader import load_implementation
from engram.runners.dynamiq_api import get_trace, management_api

console = Console()

traces_app = typer.Typer(name='traces', help='Manage workflow traces.', no_args_is_help=True)


def _fetch_all_trace_ids(jwt_env: str, app_id: str) -> list[dict]:
    """List all succeeded trace summaries for an app, paginating through all pages."""
    traces = []
    page = 1
    page_size = 100
    while True:
        resp = management_api(jwt_env, f'/apps/{app_id}/traces', {
            'page': page,
            'page_size': page_size,
            'sort': '-started_at',
        })
        data = resp.get('data', [])
        traces.extend(t for t in data if t.get('status') == 'succeeded')
        total = resp.get('pagination', {}).get('total_count', 0)
        if not data or page * page_size >= total:
            break
        page += 1
    return traces


@traces_app.command()
def pull(
    implementation: Annotated[str, typer.Argument(help='Implementation name')],
    concurrency: Annotated[int, typer.Option('--concurrency', '-c', help='Concurrent fetches')] = 10,
) -> None:
    """Pull and cache traces from hosted platform."""
    root = find_project_root()
    if root is None:
        console.print('[red]No engram.yaml found.[/red]')
        raise typer.Exit(1)

    impl_config = load_implementation(root, implementation)
    if impl_config.runner != 'dynamiq':
        console.print(f'[red]Trace pulling only supported for dynamiq runner, got {impl_config.runner}[/red]')
        raise typer.Exit(1)

    app_id = impl_config.runner_config['app_id']
    jwt_env = impl_config.config_management.jwt_env
    cache_dir = root / 'data' / 'cache'

    console.print(f'Listing traces for [cyan]{implementation}[/cyan]...')
    trace_summaries = _fetch_all_trace_ids(jwt_env, app_id)
    console.print(f'  {len(trace_summaries)} succeeded traces')

    if not trace_summaries:
        return

    cached = 0
    fetched = 0
    errors = 0

    def fetch_one(trace_id: str) -> bool:
        """Returns True if fetched from API, False if already cached."""
        trace_file = cache_dir / 'traces' / f'{trace_id}.json'
        if trace_file.exists():
            return False
        get_trace(jwt_env, trace_id, cache_dir)
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(fetch_one, t['id']): t['id'] for t in trace_summaries}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                was_fetched = future.result()
                if was_fetched:
                    fetched += 1
                else:
                    cached += 1
            except Exception as e:
                errors += 1
                console.print(f'  [red]error {tid[:12]}: {e}[/red]')

    console.print(f'[green]Done.[/green] {fetched} fetched, {cached} already cached, {errors} errors')
