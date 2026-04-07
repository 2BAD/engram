"""
Baseline tracking: workflow baselines (frozen anchors) and per-implementation references.

A workflow baseline is set once and rarely moves. It's the 'where we started' point
for the whole workflow, used as the default target for ``engram compare``.

An implementation reference is the current accepted state of one implementation. It
advances as new runs are accepted (via ``engram baseline promote``) and is used to
detect regressions within a single implementation over time.

Both live in a single ``experiments/baselines.json`` file, keyed by workflow name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engram.config.loader import load_implementation
from engram.eval.results import load_results

_BASELINES_FILENAME = 'baselines.json'


def _baselines_path(root: Path) -> Path:
    return root / 'experiments' / _BASELINES_FILENAME


def load_baselines(root: Path) -> dict[str, dict[str, Any]]:
    """Read experiments/baselines.json. Returns {} if the file does not exist."""
    path = _baselines_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_baselines(root: Path, data: dict[str, dict[str, Any]]) -> None:
    """Write experiments/baselines.json with stable, pretty-printed JSON."""
    path = _baselines_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def get_workflow_baseline(root: Path, workflow: str) -> str | None:
    """Return the experiment ID set as the baseline for ``workflow``, or None."""
    return load_baselines(root).get(workflow, {}).get('baseline')


def get_impl_reference(root: Path, workflow: str, implementation: str) -> str | None:
    """Return the experiment ID set as the reference for ``implementation`` under ``workflow``."""
    return load_baselines(root).get(workflow, {}).get('references', {}).get(implementation)


def set_workflow_baseline(root: Path, workflow: str, experiment_id: str) -> None:
    """Set the workflow baseline. Caller is responsible for verifying the experiment exists."""
    data = load_baselines(root)
    entry = data.setdefault(workflow, {})
    entry['baseline'] = experiment_id
    save_baselines(root, data)


def set_impl_reference(root: Path, workflow: str, implementation: str, experiment_id: str) -> None:
    """Set the implementation reference under its workflow. Caller verifies the experiment exists."""
    data = load_baselines(root)
    entry = data.setdefault(workflow, {})
    references = entry.setdefault('references', {})
    references[implementation] = experiment_id
    save_baselines(root, data)


def lookup_experiment(root: Path, experiment_id: str) -> tuple[str, str]:
    """
    Resolve an experiment ID to its (workflow_name, implementation_name).

    Reads ``experiments/<id>/results.json`` for the implementation, then loads the
    implementation YAML to find which workflow it belongs to.
    """
    exp_dir = root / 'experiments' / experiment_id
    if not exp_dir.exists():
        msg = f'Experiment not found: {experiment_id}'
        raise FileNotFoundError(msg)

    metadata, _results = load_results(exp_dir)
    implementation = metadata['implementation']
    impl_config = load_implementation(root, implementation)
    return impl_config.workflow, implementation
