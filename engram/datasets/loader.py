"""Load dataset inputs and labels from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engram.models.input import InputData, detect_media_type


def load_dataset_inputs(root: Path, dataset_name: str) -> list[InputData]:
    """
    Load all input files from datasets/{name}/inputs/.

    Text files are loaded as strings; images and PDFs are loaded as binary with
    the appropriate media type. Returns a list sorted by filename.
    """
    inputs_dir = root / 'datasets' / dataset_name / 'inputs'
    if not inputs_dir.exists():
        msg = f'Inputs directory not found: {inputs_dir}'
        raise FileNotFoundError(msg)

    results: list[InputData] = []
    for f in sorted(inputs_dir.iterdir()):
        if not f.is_file():
            continue
        media_type = detect_media_type(f)
        if media_type:
            results.append(InputData(filename=f.name, data=f.read_bytes(), media_type=media_type))
        else:
            results.append(InputData(filename=f.name, text=f.read_text()))
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
    if isinstance(raw, list):
        raw = _labels_array_to_dict(raw)
    if not isinstance(raw, dict):
        msg = f'labels.json must be a JSON object or array, got {type(raw).__name__}'
        raise TypeError(msg)
    return raw


def _labels_array_to_dict(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert [{filename: "x", ...labels}] to {"x": {labels}}."""
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if 'filename' not in item:
            msg = 'Each labels entry must have a "filename" field'
            raise ValueError(msg)
        labels = {k: v for k, v in item.items() if k != 'filename'}
        result[item['filename']] = labels
    return result
