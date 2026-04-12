"""Tests for the run loop, dataset loader, and results persistence."""

import json
from pathlib import Path

import pytest

import engram.eval.loop as loop_mod
from engram.datasets.loader import load_dataset_inputs, load_dataset_labels
from engram.eval.loop import run_eval as _run
from engram.eval.results import load_results, next_short_id, save_results
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

    save_results(exp_dir, 'test-exp', 1, 'classify-api', 'labeled-small', results)

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

    sampling = {'limit': 2, 'sample_seed': 7, 'source_total': 10}
    save_results(exp_dir, 'sampled-exp', 1, 'impl', 'ds', [], sampling=sampling)

    metadata, _ = load_results(exp_dir)
    assert metadata['sampling'] == sampling


# --- short_id assignment ---


def _write_results_with_short_id(root: Path, exp_id: str, short_id: int | None) -> None:
    """Write a minimal results.json under experiments/<exp_id>/ with the given short_id."""
    exp_dir = root / 'experiments' / exp_id
    exp_dir.mkdir(parents=True)
    data: dict = {
        'experiment_id': exp_id,
        'implementation': 'impl',
        'dataset': 'ds',
        'timestamp': '2026-04-04T12:00:00Z',
        'total': 0,
        'succeeded': 0,
        'failed': 0,
        'results': [],
    }
    if short_id is not None:
        data['short_id'] = short_id
    (exp_dir / 'results.json').write_text(json.dumps(data))


def test_save_results_stores_label_when_provided(tmp_path: Path):
    exp_dir = tmp_path / 'experiments' / 'labeled'
    exp_dir.mkdir(parents=True)
    save_results(exp_dir, 'labeled', 1, 'impl', 'ds', [], label='prompt-v2')
    metadata, _ = load_results(exp_dir)
    assert metadata['label'] == 'prompt-v2'


def test_save_results_strips_label_whitespace(tmp_path: Path):
    exp_dir = tmp_path / 'experiments' / 'trimmed'
    exp_dir.mkdir(parents=True)
    save_results(exp_dir, 'trimmed', 1, 'impl', 'ds', [], label='  spaced  ')
    metadata, _ = load_results(exp_dir)
    assert metadata['label'] == 'spaced'


def test_save_results_omits_label_when_none(tmp_path: Path):
    exp_dir = tmp_path / 'experiments' / 'nolabel'
    exp_dir.mkdir(parents=True)
    save_results(exp_dir, 'nolabel', 1, 'impl', 'ds', [])
    metadata, _ = load_results(exp_dir)
    assert 'label' not in metadata


def test_next_short_id_starts_at_one_for_empty_project(tmp_path: Path):
    """A project with no experiments dir returns short_id 1."""
    assert next_short_id(tmp_path) == 1


def test_next_short_id_starts_at_one_for_empty_experiments_dir(tmp_path: Path):
    """A project with an empty experiments dir also starts at 1."""
    (tmp_path / 'experiments').mkdir()
    assert next_short_id(tmp_path) == 1


def test_next_short_id_returns_max_plus_one(tmp_path: Path):
    """With existing runs carrying short_ids, returns max + 1."""
    _write_results_with_short_id(tmp_path, 'exp-a', 1)
    _write_results_with_short_id(tmp_path, 'exp-b', 5)
    _write_results_with_short_id(tmp_path, 'exp-c', 3)
    assert next_short_id(tmp_path) == 6


def test_next_short_id_ignores_runs_without_short_id(tmp_path: Path):
    """Pre-schema runs without short_id are skipped rather than renumbered."""
    _write_results_with_short_id(tmp_path, 'legacy', None)
    _write_results_with_short_id(tmp_path, 'new', 1)
    assert next_short_id(tmp_path) == 2


def test_next_short_id_skips_non_directories(tmp_path: Path):
    """Stray files in experiments/ don't break the scan."""
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments' / 'experiments.jsonl').write_text('{}\n')
    _write_results_with_short_id(tmp_path, 'exp-a', 2)
    assert next_short_id(tmp_path) == 3


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

    def configure_pricing(self, _overrides):
        pass


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

    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, sample_seed=42)
    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, sample_seed=42)
    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, sample_seed=99)

    files = capture['files']
    assert len(files[0]) == 5
    assert files[0] == files[1]  # same seed -> same subset
    assert files[0] != files[2]  # different seed -> different subset
    assert files[0] == sorted(files[0])  # subset is sorted by filename


def test_sampling_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'big', 20)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'big', concurrency=1, limit=5, sample_seed=42)

    assert capture['kwargs']['sampling'] == {'limit': 5, 'sample_seed': 42, 'source_total': 20}


def test_sampling_skipped_when_limit_exceeds_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 3)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'small', concurrency=1, limit=100, sample_seed=0)

    assert capture['kwargs']['sampling'] is None
    assert len(capture['kwargs']['results']) == 3


# --- Repeats ---


def test_repeats_runs_each_input_n_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 2)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'small', concurrency=1, repeats=3)

    results = capture['kwargs']['results']
    assert len(results) == 6
    # Grouped by input then repeat: input 0 r0, r1, r2; input 1 r0, r1, r2.
    assert [(r.input_file, r.repeat_index) for r in results] == [
        ('000.txt', 0),
        ('000.txt', 1),
        ('000.txt', 2),
        ('001.txt', 0),
        ('001.txt', 1),
        ('001.txt', 2),
    ]


def test_repeats_default_one_preserves_legacy_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 3)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    _run(tmp_path, 'impl', 'small', concurrency=1)

    results = capture['kwargs']['results']
    assert len(results) == 3
    # Every result has the default repeat_index=0 — backward compat with existing readers.
    assert all(r.repeat_index == 0 for r in results)


def test_repeats_partial_failure_keeps_failed_repeat_in_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 1)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    # Make the runner's second call (repeat_index=1) raise.
    call_count = {'n': 0}

    class _FlakyRunner:
        def snapshot_config(self, *_args, **_kwargs):
            return ConfigSnapshot(implementation='impl', platform='api', runner='stub')

        def trigger(self, content, *_args, **_kwargs):
            call_count['n'] += 1
            if call_count['n'] == 2:
                msg = 'simulated rate limit'
                raise RuntimeError(msg)
            return RunResult(input_file='', status='succeeded', output={'echo': content})

        def configure_pricing(self, _overrides):
            pass

    monkeypatch.setattr(loop_mod, 'get_runner', lambda _name: _FlakyRunner())

    _run(tmp_path, 'impl', 'small', concurrency=1, repeats=3)

    results = capture['kwargs']['results']
    assert len(results) == 3
    statuses = sorted(r.status for r in results)
    assert statuses == ['failed', 'succeeded', 'succeeded']
    failed = next(r for r in results if r.status == 'failed')
    assert failed.repeat_index == 1
    assert 'simulated rate limit' in failed.error


def test_repeats_zero_or_negative_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_dataset(tmp_path, 'small', 1)
    capture: dict = {}
    _install_runner_stubs(monkeypatch, capture)

    with pytest.raises(ValueError, match='repeats must be >= 1'):
        _run(tmp_path, 'impl', 'small', concurrency=1, repeats=0)
