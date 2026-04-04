"""Load dataset inputs and labels from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset_inputs(root: Path, dataset_name: str) -> list[tuple[str, str]]:
    """
    Load all input files from datasets/{name}/inputs/.

    Returns a list of (filename, content) tuples sorted by filename.
    """
    inputs_dir = root / 'datasets' / dataset_name / 'inputs'
    if not inputs_dir.exists():
        msg = f'Inputs directory not found: {inputs_dir}'
        raise FileNotFoundError(msg)

    results = []
    for f in sorted(inputs_dir.iterdir()):
        if f.is_file():
            results.append((f.name, f.read_text()))
    return results


def load_dataset_labels(root: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    """
    Load labels from datasets/{name}/labels.json.

    Returns a dict mapping input filename to label dict.
    Returns empty dict if no labels file exists.
    """
    labels_path = root / 'datasets' / dataset_name / 'labels.json'
    if not labels_path.exists():
        return {}

    raw = json.loads(labels_path.read_text())
    if not isinstance(raw, dict):
        msg = f'labels.json must be a JSON object, got {type(raw).__name__}'
        raise TypeError(msg)
    return raw
