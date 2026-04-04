"""Tests for the run loop, dataset loader, and results persistence."""

import json
from pathlib import Path

import pytest

from engram.datasets.loader import load_dataset_inputs, load_dataset_labels
from engram.eval.results import load_results, save_results
from engram.models.run import RunResult, TokenUsage

# --- Dataset loader ---


def test_load_dataset_inputs(tmp_path: Path):
    inputs_dir = tmp_path / 'datasets' / 'test-ds' / 'inputs'
    inputs_dir.mkdir(parents=True)
    (inputs_dir / '001.txt').write_text('hello')
    (inputs_dir / '002.txt').write_text('world')

    inputs = load_dataset_inputs(tmp_path, 'test-ds')
    assert len(inputs) == 2
    assert inputs[0] == ('001.txt', 'hello')
    assert inputs[1] == ('002.txt', 'world')


def test_load_dataset_inputs_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset_inputs(tmp_path, 'nonexistent')


def test_load_dataset_labels(tmp_path: Path):
    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    labels = {'001.txt': {'topic': 'A'}, '002.txt': {'topic': 'B'}}
    (ds_dir / 'labels.json').write_text(json.dumps(labels))

    loaded = load_dataset_labels(tmp_path, 'test-ds')
    assert loaded['001.txt']['topic'] == 'A'


def test_load_dataset_labels_missing(tmp_path: Path):
    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    assert load_dataset_labels(tmp_path, 'test-ds') == {}


# --- Results persistence ---


def test_save_and_load_results(tmp_path: Path):
    exp_dir = tmp_path / 'experiments' / 'test-exp'
    exp_dir.mkdir(parents=True)

    results = [
        RunResult(
            input_file='001.txt',
            output={'topic': 'A'},
            status='succeeded',
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost_usd=0.01,
            latency_ms=500.0,
        ),
        RunResult(
            input_file='002.txt',
            status='failed',
            error='timeout',
        ),
    ]

    save_results(exp_dir, 'test-exp', 'classify-api', 'labeled-small', results)

    assert (exp_dir / 'results.json').exists()

    metadata, loaded = load_results(exp_dir)
    assert metadata['experiment_id'] == 'test-exp'
    assert metadata['succeeded'] == 1
    assert metadata['failed'] == 1
    assert len(loaded) == 2
    assert loaded[0].output == {'topic': 'A'}
    assert loaded[0].usage.total_tokens == 150
    assert loaded[1].status == 'failed'
