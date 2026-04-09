"""Run loop: orchestrate running a workflow against a dataset."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rich.progress import Progress

from engram.config.loader import load_implementation, load_project
from engram.datasets.loader import load_dataset_inputs
from engram.eval.results import save_results
from engram.models.run import RunResult
from engram.observability.logging import log_event
from engram.observability.output_mode import get_output_mode
from engram.runners.registry import get_runner


def run_eval(
    root: Path,
    implementation_name: str,
    dataset_name: str,
    concurrency: int = 5,
    limit: int | None = None,
    seed: int = 0,
) -> str:
    """
    Run a workflow against a dataset, save results.

    If `limit` is set and smaller than the dataset size, samples that many inputs
    deterministically using `seed`. Same seed produces the same subset, so two
    runs with the same `limit` and `seed` are directly comparable.

    Returns the experiment ID.
    """
    impl_config = load_implementation(root, implementation_name)
    impl_dir = root / 'implementations' / implementation_name

    runner = get_runner(impl_config.runner)
    snapshot = runner.snapshot_config(impl_config, impl_dir)

    # Apply project-level pricing overrides (if engram.yaml is present) so the
    # runner's per-call cost calculation uses the same rates as `engram estimate`.
    pricing_overrides = load_project(root).pricing_overrides if (root / 'engram.yaml').exists() else {}
    runner.configure_pricing(pricing_overrides)

    inputs = load_dataset_inputs(root, dataset_name)

    sampling: dict | None = None
    source_total = len(inputs)
    if limit is not None and limit < source_total:
        rng = random.Random(seed)
        inputs = sorted(rng.sample(inputs, limit))
        sampling = {'limit': limit, 'seed': seed, 'source_total': source_total}
        log_event('sampling', limit=limit, seed=seed, source_total=source_total)

    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    experiment_id = f'{implementation_name}_{dataset_name}_{timestamp}'

    results: list[RunResult] = []
    mode = get_output_mode()

    def _run_single(filename: str, content: str) -> RunResult:
        result = runner.trigger(content, impl_config, impl_dir)
        result.input_file = filename
        return result

    if mode.use_rich:
        with Progress() as progress:
            task = progress.add_task(f'Running {implementation_name}', total=len(inputs))
            results = _run_concurrent(inputs, _run_single, concurrency, progress, task)
    else:
        log_event('run_start', implementation=implementation_name, dataset=dataset_name, total=len(inputs))
        results = _run_concurrent(inputs, _run_single, concurrency)
        log_event('run_complete', experiment_id=experiment_id, total=len(results))

    # Save results
    exp_dir = root / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = exp_dir / 'config-snapshot.json'
    snapshot_path.write_text(json.dumps(asdict(snapshot), indent=2))

    save_results(
        exp_dir,
        experiment_id=experiment_id,
        implementation=implementation_name,
        dataset=dataset_name,
        results=results,
        sampling=sampling,
    )

    return experiment_id


def _run_concurrent(
    inputs: list[tuple[str, str]],
    run_fn: object,
    concurrency: int,
    progress: Progress | None = None,
    task_id: object = None,
) -> list[RunResult]:
    """Run inputs concurrently and return results in input order."""
    results: dict[int, RunResult] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_fn, filename, content): i for i, (filename, content) in enumerate(inputs)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                results[idx] = RunResult(
                    input_file=inputs[idx][0],
                    status='failed',
                    error=str(e),
                )
            if progress is not None and task_id is not None:
                progress.advance(task_id)

    return [results[i] for i in range(len(inputs))]
