"""Run loop: orchestrate running a workflow against a dataset."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeRemainingColumn

from engram.config.loader import load_implementation, load_project
from engram.datasets.loader import load_dataset_inputs
from engram.eval.results import save_results
from engram.models.run import RunResult
from engram.transforms import resolve_transform

if TYPE_CHECKING:
    from engram.models.input import InputData
from engram.observability.logging import log_event
from engram.observability.output_mode import get_output_mode
from engram.runners.registry import get_runner


def run_eval(  # noqa: PLR0913, PLR0915 — top-level orchestration entry; each option maps to a CLI flag
    root: Path,
    implementation_name: str,
    dataset_name: str,
    concurrency: int = 5,
    limit: int | None = None,
    sample_seed: int = 0,
    repeats: int = 1,
    label: str | None = None,
) -> str:
    """
    Run a workflow against a dataset, save results.

    If `limit` is set and smaller than the dataset size, samples that many inputs
    deterministically using `sample_seed`. Same seed produces the same subset, so
    two runs with the same `limit` and `sample_seed` are directly comparable. The
    sample seed only controls dataset subsampling, not model sampling.

    `repeats` runs each input N times to measure the workflow's run-to-run noise.
    Each repeat is a separate API call; the total number of triggers is N * len(inputs)
    and concurrency is not scaled to compensate. Results are stored flat with a
    `repeat_index` on each RunResult, grouped by input in the saved order.

    Returns the full experiment id. Short ids (``#N``) are a local display-only
    view computed from chronology by the CLI — see ``tracking.index.compute_short_ids``.
    """
    if repeats < 1:
        msg = f'repeats must be >= 1, got {repeats}'
        raise ValueError(msg)
    impl_config = load_implementation(root, implementation_name)
    impl_dir = root / 'implementations' / implementation_name

    runner = get_runner(impl_config.runner)
    snapshot = runner.snapshot_config(impl_config, impl_dir)

    # Record transform references on the snapshot so `compare` can surface drift
    # when two experiments reshaped inputs or outputs differently.
    for side in ('input', 'output'):
        ref = getattr(impl_config.transform, side)
        if ref:
            snapshot.transform[side] = ref

    input_tx = resolve_transform(impl_config.transform.input, impl_dir) if impl_config.transform.input else None
    output_tx = resolve_transform(impl_config.transform.output, impl_dir) if impl_config.transform.output else None

    # Apply project-level pricing overrides (if engram.yaml is present) so the
    # runner's per-call cost calculation uses the same rates as `engram estimate`.
    pricing_overrides = load_project(root).pricing_overrides if (root / 'engram.yaml').exists() else {}
    runner.configure_pricing(pricing_overrides)

    inputs = load_dataset_inputs(root, dataset_name)

    sampling: dict | None = None
    source_total = len(inputs)
    if limit is not None and limit < source_total:
        rng = random.Random(sample_seed)
        inputs = sorted(rng.sample(inputs, limit), key=lambda inp: inp.filename)
        sampling = {'limit': limit, 'sample_seed': sample_seed, 'source_total': source_total}
        log_event('sampling', limit=limit, sample_seed=sample_seed, source_total=source_total)

    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    experiment_id = f'{implementation_name}_{dataset_name}_{timestamp}'

    results: list[RunResult] = []
    mode = get_output_mode()
    total_triggers = len(inputs) * repeats

    def _run_single(inp: InputData, repeat_index: int) -> RunResult:
        shaped = input_tx(inp) if input_tx else inp
        result = runner.trigger(shaped, impl_config, impl_dir)
        if output_tx and result.status == 'succeeded':
            result.output = output_tx(result.output)
        result.input_file = inp.filename
        result.repeat_index = repeat_index
        return result

    if mode.use_rich:
        columns = (
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        )
        with Progress(*columns) as progress:
            task = progress.add_task(f'Running {implementation_name}', total=total_triggers)
            results = _run_concurrent(inputs, _run_single, concurrency, repeats, progress, task)
    else:
        log_event(
            'run_start',
            implementation=implementation_name,
            dataset=dataset_name,
            total=len(inputs),
            repeats=repeats,
        )
        results = _run_concurrent(inputs, _run_single, concurrency, repeats)
        log_event('run_complete', experiment_id=experiment_id, total=len(results))

    try:
        runner.finalize(results, impl_config, impl_dir)
    except Exception as exc:  # noqa: BLE001 — finalize is best-effort enrichment; never block persistence
        log_event('finalize_failed', level='WARNING', runner=impl_config.runner, error=str(exc))

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
        label=label,
    )

    return experiment_id


def _run_concurrent(
    inputs: list[InputData],
    run_fn: Callable[..., RunResult],
    concurrency: int,
    repeats: int = 1,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[RunResult]:
    """Run inputs concurrently with N repeats per input; results are grouped by input then repeat."""
    results: dict[tuple[int, int], RunResult] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_fn, inp, r): (i, r) for i, inp in enumerate(inputs) for r in range(repeats)}

        for future in as_completed(futures):
            i, r = futures[future]
            try:
                results[(i, r)] = future.result()
            except Exception as e:  # noqa: BLE001
                results[(i, r)] = RunResult(
                    input_file=inputs[i].filename,
                    status='failed',
                    error=str(e),
                    repeat_index=r,
                )
            if progress is not None and task_id is not None:
                progress.advance(task_id)

    return [results[(i, r)] for i in range(len(inputs)) for r in range(repeats)]
