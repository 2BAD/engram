"""Tests for experiment tracking: index and comparison."""

import json
from pathlib import Path

import pytest

from engram.models.scoring import EvalReport, FieldMetrics
from engram.tracking.comparison import FieldDelta, compare_experiments, diff_config_snapshots
from engram.tracking.index import append_to_index, read_index


def _setup_experiment(root: Path, experiment_id: str, impl: str, dataset: str, topic_output: str) -> None:
    """Create a minimal experiment with results and config snapshot."""
    exp_dir = root / 'experiments' / experiment_id
    exp_dir.mkdir(parents=True)

    results_data = {
        'experiment_id': experiment_id,
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

    _setup_experiment(tmp_path, id_a, 'classify-api', 'test-ds', 'A')
    _setup_experiment(tmp_path, id_b, 'classify-api', 'test-ds', 'B')

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
def test_compare_command_prints_all_four_metric_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """engram compare renders Accuracy, Precision, Recall, and F1 tables."""
    from typer.testing import CliRunner  # noqa: PLC0415 — only needed for this CLI-level test

    from engram.cli import app  # noqa: PLC0415

    id_a, id_b = _setup_project_with_experiments(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ['compare', id_a, id_b])

    assert result.exit_code == 0
    assert 'Accuracy Comparison' in result.output
    assert 'Precision Comparison' in result.output
    assert 'Recall Comparison' in result.output
    assert 'F1 Comparison' in result.output
    # Cost table still renders.
    assert 'Cost Comparison' in result.output
    # Regressions message still triggered (accuracy 1.0 → 0.0 and F1 1.0 → 0.0).
    assert 'Regressions detected' in result.output
    assert 'topic' in result.output


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
