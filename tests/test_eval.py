"""Tests for the run loop, dataset loader, and results persistence."""

import json
from pathlib import Path

import pytest

import engram.eval.loop as loop_mod
from engram.datasets.loader import load_dataset_inputs, load_dataset_labels
from engram.eval.loop import run_eval as _run
from engram.eval.results import load_results, save_results
from engram.models.config_snapshot import ConfigSnapshot
from engram.models.implementation import ImplementationConfig
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


def test_load_dataset_labels_array_format(tmp_path: Path):
    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    labels = [
        {'filename': '001.txt', 'topic': 'A', 'sentiment': 'Positive'},
        {'filename': '002.txt', 'topic': 'B', 'sentiment': 'Negative'},
    ]
    (ds_dir / 'labels.json').write_text(json.dumps(labels))

    loaded = load_dataset_labels(tmp_path, 'test-ds')
    assert loaded['001.txt'] == {'topic': 'A', 'sentiment': 'Positive'}
    assert loaded['002.txt'] == {'topic': 'B', 'sentiment': 'Negative'}


def test_load_dataset_labels_array_missing_filename(tmp_path: Path):
    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    labels = [{'topic': 'A'}]
    (ds_dir / 'labels.json').write_text(json.dumps(labels))

    with pytest.raises(ValueError, match='filename'):
        load_dataset_labels(tmp_path, 'test-ds')


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
    assert 'sampling' not in metadata
    assert len(loaded) == 2
    assert loaded[0].output == {'topic': 'A'}
    assert loaded[0].usage.total_tokens == 150
    assert loaded[1].status == 'failed'


def test_save_results_with_sampling(tmp_path: Path):
    exp_dir = tmp_path / 'experiments' / 'sampled-exp'
    exp_dir.mkdir(parents=True)

    sampling = {'limit': 2, 'seed': 7, 'source_total': 10}
    save_results(exp_dir, 'sampled-exp', 'impl', 'ds', [], sampling=sampling)

    metadata, _ = load_results(exp_dir)
    assert metadata['sampling'] == sampling


# --- Sampling ---


def _make_dataset(root: Path, name: str, count: int) -> None:
    inputs_dir = root / 'datasets' / name / 'inputs'
    inputs_dir.mkdir(parents=True)
    for i in range(count):
        (inputs_dir / f'{i:03d}.txt').write_text(f'content-{i}')


class _StubRunner:
    def snapshot_config(self, *_args, **_kwargs):
        return ConfigSnapshot(implementation='impl', platform='api', runner='stub')

    def trigger(self, content, *_args, **_kwargs):
        return RunResult(input_file='', status='succeeded', output={'echo': content})


def _stub_load_impl(*_a, **_k) -> ImplementationConfig:
    return ImplementationConfig(workflow='wf', platform='api', runner='stub')


def _install_runner_stubs(monkeypatch: pytest.MonkeyPatch, capture: dict) -> None:
    def _stub_save(exp_dir, **kwargs):
        capture['kwargs'] = kwargs
        capture.setdefault('files', []).append([r.input_file for r in kwargs['results']])
        (exp_dir / 'results.json').write_text('{}')

    monkeypatch.setattr(loop_mod, 'load_implementation', _stub_load_impl)
    monkeypatch.setattr(loop_mod, 'get_runner', lambda _name: _StubRunner())
    monkeypatch.setattr(loop_mod, 'save_results', _stub_save)


def test_sampling_is_deterministic_with_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'big', 20)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, seed=42)
    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, seed=42)
    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, seed=99)

    files = capture['files']
    assert len(files[0]) == 5
    assert files[0] == files[1]  # same seed -> same subset
    assert files[0] != files[2]  # different seed -> different subset
    assert files[0] == sorted(files[0])  # subset is sorted by filename


def test_sampling_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'big', 20)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, seed=42)

    assert capture['kwargs']['sampling'] == {'limit': 5, 'seed': 42, 'source_total': 20}


def test_sampling_skipped_when_limit_exceeds_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 3)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'small', concurrency=1, limit=100, seed=0)

    assert capture['kwargs']['sampling'] is None
    assert len(capture['kwargs']['results']) == 3
