"""Assemble experiment data into structured context dicts for LLM analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.config.loader import load_implementation, load_workflow
from engram.datasets.loader import load_dataset_labels
from engram.eval.results import load_results
from engram.scoring.engine import score_experiment
from engram.tracking.comparison import compare_experiments, diff_config_snapshots

if TYPE_CHECKING:
    from engram.models.run import RunResult


def build_single_context(
    root: Path,
    experiment_id: str,
    max_examples: int,
) -> dict[str, Any]:
    """Assemble all relevant data for a single-experiment explanation."""
    exp_dir = root / 'experiments' / experiment_id
    metadata, results = load_results(exp_dir)

    report = score_experiment(root, experiment_id)

    snapshot = _load_snapshot(exp_dir)

    impl_config = load_implementation(root, metadata['implementation'])
    workflow = load_workflow(root, impl_config.workflow)

    labels = load_dataset_labels(root, metadata['dataset'])

    examples = _sample_examples(results, labels, workflow.scorers, max_examples)

    return {
        'mode': 'single',
        'experiment_id': experiment_id,
        'implementation': metadata['implementation'],
        'dataset': metadata['dataset'],
        'timestamp': metadata.get('timestamp', ''),
        'label': metadata.get('label'),
        'model': snapshot.get('models', ['unknown'])[0] if snapshot.get('models') else 'unknown',
        'prompts': snapshot.get('prompts', {}),
        'runner_config': snapshot.get('runner_config', {}),
        'workflow_description': workflow.description,
        'output_fields': {
            name: {'type': f.type, 'values': f.values, 'description': f.description}
            for name, f in workflow.output_fields.items()
        },
        'field_metrics': [asdict(fm) for fm in report.field_metrics],
        'confusion_matrices': [asdict(cm) for cm in report.confusion_matrices],
        'cost': {
            'total': report.cost_total_usd,
            'avg': report.cost_avg_usd,
            'median': report.cost_median_usd,
            'p95': report.cost_p95_usd,
        },
        'total_examples': metadata.get('total', 0),
        'succeeded': metadata.get('succeeded', 0),
        'failed': metadata.get('failed', 0),
        'examples': examples,
    }


def build_comparison_context(
    root: Path,
    id_a: str,
    id_b: str,
    max_examples: int,
) -> dict[str, Any]:
    """Assemble context for a two-experiment comparison explanation."""
    half = max(1, max_examples // 2)
    ctx_a = build_single_context(root, id_a, max_examples=half)
    ctx_b = build_single_context(root, id_b, max_examples=half)

    comparison = compare_experiments(root, id_a, id_b)
    config_diff = diff_config_snapshots(root, id_a, id_b, show_prompts=True)

    return {
        'mode': 'comparison',
        'experiment_a': ctx_a,
        'experiment_b': ctx_b,
        'field_deltas': {
            name: {
                'accuracy_delta': d.accuracy_delta,
                'f1_delta': d.f1_delta,
                'precision_delta': d.precision_delta,
                'recall_delta': d.recall_delta,
                'regressed': d.regressed,
            }
            for name, d in comparison.field_deltas.items()
        },
        'cost_a': comparison.cost_a,
        'cost_b': comparison.cost_b,
        'regressions': comparison.regressions,
        'config_diff': config_diff,
    }


def _sample_examples(
    results: list[RunResult],
    labels: dict[str, dict[str, Any]],
    scorers: dict[str, str],
    max_examples: int,
) -> list[dict[str, Any]]:
    """
    Sample examples prioritizing diagnostic value.

    Order: runner failures first, then scoring mismatches, then correct examples.
    """
    failed_runs: list[RunResult] = []
    mismatches: list[RunResult] = []
    correct: list[RunResult] = []

    for r in results:
        if r.status != 'succeeded':
            failed_runs.append(r)
            continue

        example_labels = labels.get(r.input_file, {})
        if not example_labels:
            correct.append(r)
            continue

        has_error = any(
            field_name in example_labels and r.output.get(field_name) != example_labels[field_name]
            for field_name in scorers
        )
        (mismatches if has_error else correct).append(r)

    sampled: list[RunResult] = []
    for group in (failed_runs, mismatches, correct):
        remaining = max_examples - len(sampled)
        if remaining <= 0:
            break
        sampled.extend(group[:remaining])

    examples: list[dict[str, Any]] = []
    for r in sampled:
        ex: dict[str, Any] = {
            'input_file': r.input_file,
            'status': r.status,
            'output': r.output,
        }
        if r.input_file in labels:
            ex['expected'] = labels[r.input_file]
        if r.error:
            ex['error'] = r.error
        examples.append(ex)

    return examples


def _load_snapshot(exp_dir: Path) -> dict[str, Any]:
    """Load a config snapshot from an experiment directory."""
    path = exp_dir / 'config-snapshot.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text())
