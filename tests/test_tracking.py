"""Tests for experiment tracking: index and comparison."""

import json
from pathlib import Path

import pytest

from engram.models.scoring import EvalReport, FieldMetrics
from engram.tracking.comparison import FieldDelta, compare_experiments, diff_config_snapshots
from engram.tracking.index import append_to_index, list_experiments, read_index, resolve_experiment_id


def _setup_experiment(
    root: Path,
    experiment_id: str,
    impl: str,
    dataset: str,
    topic_output: str,
    short_id: int = 1,
) -> None:
    """Create a minimal experiment with results and config snapshot."""
    exp_dir = root / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)

    results_data = {
        'experiment_id': experiment_id,
        'short_id': short_id,
        'implementation': impl,
        'dataset': dataset,
        'timestamp': '2026-04-04T12:00:00Z',
        'total': 1,
        'succeeded': 1,
        'failed': 0,
        'results': [
            {
                'input_file': '001.txt',
                'output': {'topic': topic_output},
                'status': 'succeeded',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'cost_usd': 0.01,
                'latency_ms': 500,
                'error': '',
            }
        ],
    }
    (exp_dir / 'results.json').write_text(json.dumps(results_data))
    (exp_dir / 'config-snapshot.json').write_text(
        json.dumps({'models': ['claude-sonnet'], 'prompts': {'system.md': 'Classify.'}, 'runner_config': {}})
    )


def _setup_project_with_experiments(tmp_path: Path) -> tuple[str, str]:
    """Create a project with two experiments for comparison."""
    (tmp_path / 'engram.yaml').write_text('name: test\n')

    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(
        'name: classify\n'
        'output:\n'
        '  fields:\n'
        '    topic:\n'
        '      type: enum\n'
        '      values: [A, B]\n'
        'scorers:\n'
        '  topic: exact_match\n'
    )

    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text('workflow: classify\nplatform: api\nrunner: anthropic\n')

    ds_dir = tmp_path / 'datasets' / 'test-ds'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text('name: test-ds\n')
    (ds_dir / 'labels.json').write_text(json.dumps({'001.txt': {'topic': 'A'}}))

    (tmp_path / 'experiments').mkdir(exist_ok=True)

    id_a = 'classify-api_test-ds_20260404_120000'
    id_b = 'classify-api_test-ds_20260404_130000'

    _setup_experiment(tmp_path, id_a, 'classify-api', 'test-ds', 'A', short_id=1)
    _setup_experiment(tmp_path, id_b, 'classify-api', 'test-ds', 'B', short_id=2)

    return id_a, id_b


# --- Index ---


def test_append_and_read_index(tmp_path: Path):
    (tmp_path / 'experiments').mkdir()

    # Create a minimal experiment for the index to read
    exp_id = 'test-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': 1,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 1,
                'succeeded': 1,
                'failed': 0,
                'results': [],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    report = EvalReport(
        experiment_id=exp_id,
        field_metrics=[
            FieldMetrics(
                field_name='topic',
                accuracy=0.95,
                precision=0.92,
                recall=0.9,
                f1=0.91,
                total=100,
                correct=95,
            )
        ],
        cost_total_usd=1.23,
        cost_avg_usd=0.0123,
    )

    append_to_index(tmp_path, report)

    entries = read_index(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry['id'] == exp_id
    assert entry['macro_accuracy'] == 0.95
    assert entry['macro_f1'] == 0.91
    assert entry['field_accuracy'] == {'topic': 0.95}
    assert entry['field_precision'] == {'topic': 0.92}
    assert entry['field_recall'] == {'topic': 0.9}
    assert entry['field_f1'] == {'topic': 0.91}
    assert entry['models'] == ['claude-sonnet']


def test_read_index_empty(tmp_path: Path):
    assert read_index(tmp_path) == []


def test_append_to_index_records_short_id(tmp_path: Path):
    """append_to_index copies short_id from results.json metadata into the index row."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-a', 'classify-api', 'test-ds', 'A', short_id=42)
    append_to_index(tmp_path, EvalReport(experiment_id='exp-a', field_metrics=[]))
    assert read_index(tmp_path)[0]['short_id'] == 42


def test_append_to_index_carries_label(tmp_path: Path):
    """A label in results.json metadata is copied to the index row."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-labeled', 'classify-api', 'test-ds', 'A', short_id=1)
    results_path = tmp_path / 'experiments' / 'exp-labeled' / 'results.json'
    data = json.loads(results_path.read_text())
    data['label'] = 'prompt-v2'
    results_path.write_text(json.dumps(data))

    append_to_index(tmp_path, EvalReport(experiment_id='exp-labeled', field_metrics=[]))
    assert read_index(tmp_path)[0]['label'] == 'prompt-v2'


def test_append_to_index_omits_label_when_absent(tmp_path: Path):
    """Index rows don't get an empty or null label field when metadata has none."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-nolabel', 'classify-api', 'test-ds', 'A', short_id=1)
    append_to_index(tmp_path, EvalReport(experiment_id='exp-nolabel', field_metrics=[]))
    assert 'label' not in read_index(tmp_path)[0]


def test_append_to_index_carries_labels_hash(tmp_path: Path):
    """The labels_hash from a scored report is mirrored into the index summary."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-hash', 'classify-api', 'test-ds', 'A', short_id=1)
    report = EvalReport(experiment_id='exp-hash', field_metrics=[], labels_hash='f' * 64)
    append_to_index(tmp_path, report)
    assert read_index(tmp_path)[0]['labels_hash'] == 'f' * 64


def test_append_to_index_omits_labels_hash_when_absent(tmp_path: Path):
    """Reports with no labels_hash (older runs) don't add an empty field to the summary."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-nohash', 'classify-api', 'test-ds', 'A', short_id=1)
    append_to_index(tmp_path, EvalReport(experiment_id='exp-nohash', field_metrics=[]))
    assert 'labels_hash' not in read_index(tmp_path)[0]


# --- resolve_experiment_id ---


def test_resolve_non_numeric_returns_input_unchanged(tmp_path: Path):
    """Anything not a pure integer string passes through untouched — it's a full id."""
    (tmp_path / 'experiments').mkdir()
    assert resolve_experiment_id(tmp_path, 'classify-anthropic_sample_20260412_123456') == (
        'classify-anthropic_sample_20260412_123456'
    )


def test_resolve_short_id_against_index(tmp_path: Path):
    """Numeric input hits the index first and returns the matching full id."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-a', 'classify-api', 'test-ds', 'A', short_id=1)
    _setup_experiment(tmp_path, 'exp-b', 'classify-api', 'test-ds', 'B', short_id=2)
    append_to_index(tmp_path, EvalReport(experiment_id='exp-a', field_metrics=[]))
    append_to_index(tmp_path, EvalReport(experiment_id='exp-b', field_metrics=[]))

    assert resolve_experiment_id(tmp_path, '#2') == 'exp-b'


def test_resolve_short_id_falls_back_to_experiments_dir(tmp_path: Path):
    """Unscored runs not in the index are still reachable by short_id via the dir scan."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-unscored', 'classify-api', 'test-ds', 'A', short_id=7)
    # Note: no append_to_index — this experiment has never been scored.
    assert resolve_experiment_id(tmp_path, '#7') == 'exp-unscored'


def test_resolve_short_id_not_found(tmp_path: Path):
    """Looking up a short_id that doesn't exist raises FileNotFoundError with a clear message."""
    (tmp_path / 'experiments').mkdir()
    _setup_experiment(tmp_path, 'exp-a', 'classify-api', 'test-ds', 'A', short_id=1)

    with pytest.raises(FileNotFoundError, match='No experiment found with short_id #99'):
        resolve_experiment_id(tmp_path, '#99')


# --- @ / @~N recency resolution ---


def _setup_experiment_with_timestamp(
    root: Path,
    experiment_id: str,
    impl: str,
    dataset: str,
    timestamp: str,
    short_id: int,
) -> None:
    """Create a minimal results.json with a specific timestamp for recency tests."""
    exp_dir = root / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': experiment_id,
                'short_id': short_id,
                'implementation': impl,
                'dataset': dataset,
                'timestamp': timestamp,
                'total': 0,
                'succeeded': 0,
                'failed': 0,
                'results': [],
            }
        )
    )


def test_list_experiments_sorted_newest_first(tmp_path: Path):
    """list_experiments returns all experiments sorted by timestamp descending."""
    _setup_experiment_with_timestamp(tmp_path, 'old', 'a', 'ds', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'new', 'a', 'ds', '2026-04-10T00:00:00', 2)
    _setup_experiment_with_timestamp(tmp_path, 'mid', 'a', 'ds', '2026-04-05T00:00:00', 3)

    entries = list_experiments(tmp_path)
    assert [e['experiment_id'] for e in entries] == ['new', 'mid', 'old']


def test_list_experiments_filters(tmp_path: Path):
    """impl and dataset filters narrow the result set by exact match."""
    _setup_experiment_with_timestamp(tmp_path, 'ant-sample', 'anthropic', 'sample', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'oai-sample', 'openai', 'sample', '2026-04-02T00:00:00', 2)
    _setup_experiment_with_timestamp(tmp_path, 'ant-full', 'anthropic', 'full', '2026-04-03T00:00:00', 3)

    ant_only = list_experiments(tmp_path, impl='anthropic')
    assert [e['experiment_id'] for e in ant_only] == ['ant-full', 'ant-sample']

    sample_only = list_experiments(tmp_path, dataset='sample')
    assert [e['experiment_id'] for e in sample_only] == ['oai-sample', 'ant-sample']

    ant_sample = list_experiments(tmp_path, impl='anthropic', dataset='sample')
    assert [e['experiment_id'] for e in ant_sample] == ['ant-sample']


def test_resolve_at_returns_newest(tmp_path: Path):
    """@ returns the single most recent experiment by timestamp."""
    _setup_experiment_with_timestamp(tmp_path, 'old', 'a', 'ds', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'new', 'a', 'ds', '2026-04-10T00:00:00', 2)

    assert resolve_experiment_id(tmp_path, '@') == 'new'


def test_resolve_at_dash_n_walks_back(tmp_path: Path):
    """@~N returns the (N+1)th most recent experiment (0-indexed offset from the newest)."""
    _setup_experiment_with_timestamp(tmp_path, 'third', 'a', 'ds', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'second', 'a', 'ds', '2026-04-02T00:00:00', 2)
    _setup_experiment_with_timestamp(tmp_path, 'first', 'a', 'ds', '2026-04-03T00:00:00', 3)

    assert resolve_experiment_id(tmp_path, '@') == 'first'
    assert resolve_experiment_id(tmp_path, '@~0') == 'first'  # @ == @-0
    assert resolve_experiment_id(tmp_path, '@~1') == 'second'
    assert resolve_experiment_id(tmp_path, '@~2') == 'third'


def test_resolve_at_with_impl_filter(tmp_path: Path):
    """@ with impl filter returns the newest matching that implementation, ignoring others."""
    _setup_experiment_with_timestamp(tmp_path, 'ant-old', 'anthropic', 'sample', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'oai-newest', 'openai', 'sample', '2026-04-10T00:00:00', 2)
    _setup_experiment_with_timestamp(tmp_path, 'ant-new', 'anthropic', 'sample', '2026-04-05T00:00:00', 3)

    assert resolve_experiment_id(tmp_path, '@', impl='anthropic') == 'ant-new'
    assert resolve_experiment_id(tmp_path, '@', impl='openai') == 'oai-newest'


def test_resolve_at_with_dataset_filter(tmp_path: Path):
    """@ with dataset filter narrows to that dataset only."""
    _setup_experiment_with_timestamp(tmp_path, 'sample-old', 'a', 'sample', '2026-04-01T00:00:00', 1)
    _setup_experiment_with_timestamp(tmp_path, 'full-new', 'a', 'full', '2026-04-10T00:00:00', 2)

    assert resolve_experiment_id(tmp_path, '@', dataset='sample') == 'sample-old'


def test_resolve_at_empty_project_errors(tmp_path: Path):
    """@ on a project with zero experiments exits with a clear message."""
    with pytest.raises(FileNotFoundError, match=r'No experiments.*to resolve @'):
        resolve_experiment_id(tmp_path, '@')


def test_resolve_at_out_of_range_errors(tmp_path: Path):
    """@~N beyond the available experiments reports the real count in the error."""
    _setup_experiment_with_timestamp(tmp_path, 'only', 'a', 'ds', '2026-04-01T00:00:00', 1)

    with pytest.raises(FileNotFoundError, match='@~5 is out of range: only 1 experiment'):
        resolve_experiment_id(tmp_path, '@~5')


def test_resolve_at_empty_scope_mentions_filters(tmp_path: Path):
    """When filters knock out every experiment, the error names the filters so the user can debug."""
    _setup_experiment_with_timestamp(tmp_path, 'exists', 'anthropic', 'sample', '2026-04-01T00:00:00', 1)

    with pytest.raises(FileNotFoundError, match=r'impl=openai'):
        resolve_experiment_id(tmp_path, '@', impl='openai')


def test_format_ref_medium_with_label():
    """format_ref_medium appends the label in brackets when present."""
    from engram.display.experiment_ref import format_ref_medium  # noqa: PLC0415

    entry = {'short_id': 7, 'implementation': 'anthropic', 'dataset': 'sample', 'label': 'prompt-v2'}
    assert format_ref_medium(entry) == '#7 anthropic/sample [prompt-v2]'


def test_format_ref_medium_without_label():
    """format_ref_medium omits the bracket suffix when label is absent."""
    from engram.display.experiment_ref import format_ref_medium  # noqa: PLC0415

    entry = {'short_id': 7, 'implementation': 'anthropic', 'dataset': 'sample'}
    assert format_ref_medium(entry) == '#7 anthropic/sample'


def test_format_ref_medium_returns_plain_text_with_markup_chars():
    """Labels containing Rich markup characters are returned as plain text (escaping is caller's job)."""
    from engram.display.experiment_ref import format_ref_medium  # noqa: PLC0415

    entry = {'short_id': 1, 'implementation': 'a', 'dataset': 'b', 'label': '[bold]test'}
    ref = format_ref_medium(entry)
    assert '[bold]test' in ref


def test_resolve_short_id_ignores_impl_filter(tmp_path: Path):
    """short_id is globally unique, so impl/dataset filters are ignored for numeric lookups."""
    _setup_experiment_with_timestamp(tmp_path, 'exp-ant', 'anthropic', 'sample', '2026-04-01T00:00:00', 1)

    # The lookup succeeds even though the filter would exclude the only match.
    assert resolve_experiment_id(tmp_path, '#1', impl='openai') == 'exp-ant'


def test_append_to_index_upserts_rescored_experiment(tmp_path: Path):
    """Re-scoring an experiment replaces its entry in place, not as a duplicate."""
    (tmp_path / 'experiments').mkdir()

    _setup_experiment(tmp_path, 'exp-a', 'classify-api', 'test-ds', 'A', short_id=1)
    _setup_experiment(tmp_path, 'exp-b', 'classify-api', 'test-ds', 'B', short_id=2)
    _setup_experiment(tmp_path, 'exp-c', 'classify-api', 'test-ds', 'A', short_id=3)

    def _report(exp_id: str, accuracy: float) -> EvalReport:
        return EvalReport(
            experiment_id=exp_id,
            field_metrics=[
                FieldMetrics(
                    field_name='topic',
                    accuracy=accuracy,
                    precision=accuracy,
                    recall=accuracy,
                    f1=accuracy,
                    total=1,
                    correct=int(accuracy),
                )
            ],
        )

    append_to_index(tmp_path, _report('exp-a', 1.0))
    append_to_index(tmp_path, _report('exp-b', 0.5))
    append_to_index(tmp_path, _report('exp-c', 0.0))

    # Re-score exp-b with new metrics: should replace in place, not duplicate.
    append_to_index(tmp_path, _report('exp-b', 1.0))

    entries = read_index(tmp_path)
    assert [e['id'] for e in entries] == ['exp-a', 'exp-b', 'exp-c']
    assert entries[1]['macro_accuracy'] == 1.0


def test_index_records_avg_output_tokens(tmp_path: Path):
    """Mean completion tokens across runs with a recorded response land in the index."""
    (tmp_path / 'experiments').mkdir()
    exp_id = 'test-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': 7,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 3,
                'succeeded': 2,
                'failed': 1,
                'results': [
                    {
                        'input_file': '001.txt',
                        'output': {'topic': 'A'},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140},
                        'cost_usd': 0.01,
                        'latency_ms': 500,
                        'error': '',
                    },
                    {
                        'input_file': '002.txt',
                        'output': {'topic': 'B'},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 60, 'total_tokens': 160},
                        'cost_usd': 0.01,
                        'latency_ms': 500,
                        'error': '',
                    },
                    # API-error path: no tokens recorded, excluded from calibration.
                    {
                        'input_file': '003.txt',
                        'output': {},
                        'status': 'failed',
                        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                        'cost_usd': 0.0,
                        'latency_ms': 100,
                        'error': 'api error',
                    },
                ],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    append_to_index(tmp_path, EvalReport(experiment_id=exp_id, field_metrics=[]))

    entries = read_index(tmp_path)
    assert entries[0]['avg_output_tokens'] == 50  # mean of 40 and 60


def test_index_omits_avg_output_tokens_when_no_data(tmp_path: Path):
    """When every run failed before tokens were recorded, the field is omitted."""
    (tmp_path / 'experiments').mkdir()
    exp_id = 'empty-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': 8,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 0,
                'succeeded': 0,
                'failed': 0,
                'results': [],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    append_to_index(tmp_path, EvalReport(experiment_id=exp_id, field_metrics=[]))

    entries = read_index(tmp_path)
    assert 'avg_output_tokens' not in entries[0]


def test_index_records_repeat_aware_metrics(tmp_path: Path):
    """A multi-repeat experiment writes the four new per-field metrics and the repeats count."""
    (tmp_path / 'experiments').mkdir()
    exp_id = 'repeat-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': 9,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 4,
                'succeeded': 4,
                'failed': 0,
                'results': [
                    {
                        'input_file': '001.txt',
                        'output': {'topic': 'A'},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                        'cost_usd': 0.0,
                        'latency_ms': 100,
                        'error': '',
                        'repeat_index': r,
                    }
                    for r in range(4)
                ],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    report = EvalReport(
        experiment_id=exp_id,
        field_metrics=[
            FieldMetrics(
                field_name='topic',
                accuracy=0.85,
                precision=0.85,
                recall=0.85,
                f1=0.85,
                total=4,
                correct=3,
                accuracy_stdev=0.05,
                mean_agreement_rate=0.95,
                majority_rate=1.0,
                fleiss_kappa=0.78,
            )
        ],
    )

    append_to_index(tmp_path, report)
    entry = read_index(tmp_path)[0]

    assert entry['repeats'] == 4
    assert entry['field_accuracy_stdev'] == {'topic': 0.05}
    assert entry['field_mean_agreement_rate'] == {'topic': 0.95}
    assert entry['field_majority_rate'] == {'topic': 1.0}
    assert entry['field_fleiss_kappa'] == {'topic': 0.78}


def test_index_omits_repeat_aware_metrics_for_single_repeat(tmp_path: Path):
    """Single-repeat experiments must not gain the new schema keys (D12 backward compat)."""
    (tmp_path / 'experiments').mkdir()
    exp_id = 'single-exp'
    exp_dir = tmp_path / 'experiments' / exp_id
    exp_dir.mkdir()
    (exp_dir / 'results.json').write_text(
        json.dumps(
            {
                'experiment_id': exp_id,
                'short_id': 10,
                'implementation': 'classify-api',
                'dataset': 'test-ds',
                'timestamp': '2026-04-04T12:00:00Z',
                'total': 1,
                'succeeded': 1,
                'failed': 0,
                'results': [
                    {
                        'input_file': '001.txt',
                        'output': {'topic': 'A'},
                        'status': 'succeeded',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                        'cost_usd': 0.0,
                        'latency_ms': 100,
                        'error': '',
                    }
                ],
            }
        )
    )
    (exp_dir / 'config-snapshot.json').write_text(json.dumps({'models': ['claude-sonnet']}))

    report = EvalReport(
        experiment_id=exp_id,
        field_metrics=[
            FieldMetrics(
                field_name='topic',
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                total=1,
                correct=1,
            )
        ],
    )

    append_to_index(tmp_path, report)
    entry = read_index(tmp_path)[0]

    assert 'repeats' not in entry
    assert 'field_accuracy_stdev' not in entry
    assert 'field_mean_agreement_rate' not in entry
    assert 'field_majority_rate' not in entry
    assert 'field_fleiss_kappa' not in entry


# --- Field Delta ---


def test_field_delta():
    # Regression is gated on F1, not accuracy. Compute all four deltas independently.
    delta = FieldDelta(
        field_name='topic',
        accuracy_a=0.8,
        accuracy_b=0.9,
        precision_a=0.75,
        precision_b=0.85,
        recall_a=0.8,
        recall_b=0.9,
        f1_a=0.77,
        f1_b=0.87,
    )
    assert delta.accuracy_delta == pytest.approx(0.1)
    assert delta.precision_delta == pytest.approx(0.1)
    assert delta.recall_delta == pytest.approx(0.1)
    assert delta.f1_delta == pytest.approx(0.1)
    assert delta.delta == pytest.approx(0.1)  # delta aliases f1_delta
    assert not delta.regressed

    delta_down = FieldDelta(
        field_name='sentiment',
        accuracy_a=0.9,
        accuracy_b=0.7,
        f1_a=0.88,
        f1_b=0.65,
    )
    assert delta_down.f1_delta == pytest.approx(-0.23)
    assert delta_down.regressed


def test_field_delta_accuracy_up_but_f1_down_is_regression():
    """A run where accuracy rises but F1 drops (class imbalance shift) counts as a regression."""
    delta = FieldDelta(
        field_name='topic',
        accuracy_a=0.7,
        accuracy_b=0.75,  # accuracy up
        f1_a=0.65,
        f1_b=0.5,  # but F1 down — rare classes got worse
    )
    assert delta.accuracy_delta > 0
    assert delta.f1_delta < 0
    assert delta.regressed  # gated on F1


# --- Config diff ---


def test_diff_config_snapshots(tmp_path: Path):
    id_a, id_b = 'exp-a', 'exp-b'

    exp_a = tmp_path / 'experiments' / id_a
    exp_b = tmp_path / 'experiments' / id_b
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)

    (exp_a / 'config-snapshot.json').write_text(
        json.dumps({'models': ['gpt-4'], 'prompts': {'system.md': 'v1'}, 'runner_config': {'max_tokens': '1000'}})
    )
    (exp_b / 'config-snapshot.json').write_text(
        json.dumps({'models': ['gpt-4.1'], 'prompts': {'system.md': 'v2'}, 'runner_config': {'max_tokens': '2000'}})
    )

    lines = diff_config_snapshots(tmp_path, id_a, id_b)
    assert any('Models' in line for line in lines)
    assert any('max_tokens' in line for line in lines)
    assert any('system.md' in line for line in lines)


def test_diff_config_snapshots_surfaces_transform_drift(tmp_path: Path):
    id_a, id_b = 'exp-a', 'exp-b'
    exp_a = tmp_path / 'experiments' / id_a
    exp_b = tmp_path / 'experiments' / id_b
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)

    (exp_a / 'config-snapshot.json').write_text(json.dumps({'transform': {'input': 'transforms.v1'}}))
    (exp_b / 'config-snapshot.json').write_text(
        json.dumps({'transform': {'input': 'transforms.v2', 'output': 'transforms.normalize'}})
    )

    lines = diff_config_snapshots(tmp_path, id_a, id_b)
    assert any('transform.input' in line and 'v1' in line and 'v2' in line for line in lines)
    assert any('transform.output' in line and 'normalize' in line for line in lines)


# --- Comparison ---


def test_compare_experiments(tmp_path: Path):
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    result = compare_experiments(tmp_path, id_a, id_b)

    assert result.experiment_a == id_a
    assert result.experiment_b == id_b
    assert 'topic' in result.field_deltas

    delta = result.field_deltas['topic']
    # A got topic right (A=A), B got it wrong (B!=A)
    assert delta.accuracy_a == 1.0
    assert delta.accuracy_b == 0.0
    # F1 mirrors accuracy in this simple one-label-each case.
    assert delta.f1_a == 1.0
    assert delta.f1_b == 0.0
    # enum + exact_match → flagged as classification in the delta so display renders real numbers.
    assert delta.is_classification is True
    assert result.regressions == ['topic']


@pytest.mark.usefixtures('rich_mode')
def test_compare_command_prints_per_field_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """engram compare renders one table per field with all metrics as rows."""
    from typer.testing import CliRunner  # noqa: PLC0415 — only needed for this CLI-level test

    from engram.cli import app  # noqa: PLC0415

    id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner(env={'COLUMNS': '200'})
    result = runner.invoke(app, ['compare', id_a, id_b])

    assert result.exit_code == 0
    # Per-field table uses the field name as title and contains all metric rows.
    assert 'topic' in result.output
    assert 'Accuracy' in result.output
    assert 'Precision' in result.output
    assert 'Recall' in result.output
    assert 'F1' in result.output
    # Cost table still renders.
    assert 'Cost Comparison' in result.output
    # Regressions message still triggered.
    assert 'Regressions detected' in result.output
    # Headers show the pretty refs (#N impl/dataset), not the long full ids.
    assert '#1' in result.output
    assert '#2' in result.output
    assert 'classify-api/test-ds' in result.output
    assert id_a not in result.output
    assert id_b not in result.output


def test_compare_command_json_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With --json, compare emits a structured dict with field deltas, cost, config changes, and regressions."""
    from typer.testing import CliRunner  # noqa: PLC0415

    from engram.cli import app  # noqa: PLC0415

    id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ['--json', 'compare', id_a, id_b])

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload['experiment_a'] == id_a
    assert payload['experiment_b'] == id_b
    assert 'topic' in payload['field_deltas']
    topic_delta = payload['field_deltas']['topic']
    assert topic_delta['f1_a'] == 1.0
    assert topic_delta['f1_b'] == 0.0
    assert topic_delta['is_classification'] is True
    assert 'config_changes' in payload
    assert payload['regressions'] == ['topic']


def _write_eval_json(root: Path, experiment_id: str, labels_hash: str, labels_count: int, scored_at: str) -> None:
    """Pre-seed a saved eval report so compare_experiments reads it instead of re-scoring."""
    (root / 'experiments' / experiment_id / 'eval.json').write_text(
        json.dumps(
            {
                'experiment_id': experiment_id,
                'matched_examples': 1,
                'field_metrics': [],
                'confusion_matrices': [],
                'cost_total_usd': 0.0,
                'cost_avg_usd': 0.0,
                'cost_median_usd': 0.0,
                'cost_p95_usd': 0.0,
                'labels_hash': labels_hash,
                'labels_count': labels_count,
                'labels_scored_at': scored_at,
            }
        )
    )


def test_compare_experiments_populates_labels_fingerprint(tmp_path: Path):
    """ComparisonResult carries the labels_hash/count/scored_at from each saved report."""
    id_a, id_b = _setup_project_with_experiments(tmp_path)
    _write_eval_json(tmp_path, id_a, 'a' * 64, 50, '2026-01-01T00:00:00+00:00')
    _write_eval_json(tmp_path, id_b, 'b' * 64, 52, '2026-03-14T00:00:00+00:00')

    result = compare_experiments(tmp_path, id_a, id_b)
    assert result.labels_a == {'hash': 'a' * 64, 'count': 50, 'scored_at': '2026-01-01T00:00:00+00:00'}
    assert result.labels_b == {'hash': 'b' * 64, 'count': 52, 'scored_at': '2026-03-14T00:00:00+00:00'}


@pytest.mark.usefixtures('rich_mode')
def test_compare_command_warns_on_label_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rich compare output warns when two same-dataset experiments have different labels_hash."""
    from typer.testing import CliRunner  # noqa: PLC0415

    from engram.cli import app  # noqa: PLC0415

    id_a, id_b = _setup_project_with_experiments(tmp_path)
    _write_eval_json(tmp_path, id_a, 'a' * 64, 50, '2026-01-01T00:00:00+00:00')
    _write_eval_json(tmp_path, id_b, 'b' * 64, 52, '2026-03-14T00:00:00+00:00')

    monkeypatch.chdir(tmp_path)
    result = CliRunner(env={'COLUMNS': '200'}).invoke(app, ['compare', id_a, id_b])

    assert result.exit_code == 0
    assert 'label set differs' in result.output
    assert 'aaaaaaaaaaaa' in result.output  # 12-char truncated hash_a
    assert 'bbbbbbbbbbbb' in result.output  # 12-char truncated hash_b


@pytest.mark.usefixtures('rich_mode')
def test_compare_command_no_warning_when_labels_hash_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No drift warning when both experiments were scored against the same labels payload."""
    from typer.testing import CliRunner  # noqa: PLC0415

    from engram.cli import app  # noqa: PLC0415

    id_a, id_b = _setup_project_with_experiments(tmp_path)
    same_hash = 'c' * 64
    _write_eval_json(tmp_path, id_a, same_hash, 50, '2026-01-01T00:00:00+00:00')
    _write_eval_json(tmp_path, id_b, same_hash, 50, '2026-03-14T00:00:00+00:00')

    monkeypatch.chdir(tmp_path)
    result = CliRunner(env={'COLUMNS': '200'}).invoke(app, ['compare', id_a, id_b])

    assert result.exit_code == 0
    assert 'label set differs' not in result.output
