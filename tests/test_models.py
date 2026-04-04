"""Tests for core data models."""

import json
from dataclasses import asdict

from engram.models import (
    ConfigManagement,
    ConfigSnapshot,
    DatasetConfig,
    DatasetEntry,
    EvalReport,
    Experiment,
    ExperimentSummary,
    FieldMetrics,
    ImplementationConfig,
    OutputField,
    ProjectConfig,
    RunResult,
    TokenUsage,
    WorkflowConfig,
)


def test_project_config():
    config = ProjectConfig(name='test', description='a test project')
    assert config.name == 'test'
    d = asdict(config)
    assert d['name'] == 'test'
    assert d['pricing_overrides'] == {}


def test_workflow_config():
    wf = WorkflowConfig(
        name='classify',
        output_fields={
            'topic': OutputField(type='enum', values=['A', 'B']),
            'confidence': OutputField(type='number'),
        },
        scorers={'topic': 'exact_match', 'confidence': 'numeric_tolerance(0.1)'},
        confusion_matrices=['topic'],
    )
    assert wf.name == 'classify'
    assert wf.output_fields['topic'].values == ['A', 'B']
    assert wf.scorers['confidence'] == 'numeric_tolerance(0.1)'


def test_implementation_config():
    impl = ImplementationConfig(
        workflow='classify',
        platform='api',
        runner='anthropic',
        runner_config={'api_key_env': 'ANTHROPIC_API_KEY', 'model': 'claude-sonnet-4-5-20250514'},
        config_management=ConfigManagement(mode='local'),
    )
    assert impl.platform == 'api'
    assert impl.runner_config['model'] == 'claude-sonnet-4-5-20250514'


def test_dataset_entry():
    entry = DatasetEntry(input_file='001.txt', input_data='hello world', labels={'topic': 'A'})
    assert entry.labels['topic'] == 'A'


def test_dataset_config():
    ds = DatasetConfig(name='labeled-small', description='small labeled set')
    assert ds.name == 'labeled-small'


def test_run_result_defaults():
    result = RunResult(input_file='001.txt')
    assert result.status == 'succeeded'
    assert result.cost_usd == 0.0
    assert result.usage.total_tokens == 0


def test_run_result_serialization():
    result = RunResult(
        input_file='001.txt',
        output={'topic': 'A'},
        status='succeeded',
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cost_usd=0.01,
        latency_ms=1200.5,
    )
    d = asdict(result)
    serialized = json.dumps(d)
    loaded = json.loads(serialized)
    assert loaded['output']['topic'] == 'A'
    assert loaded['usage']['total_tokens'] == 150


def test_experiment():
    exp = Experiment(
        id='classify-api_labeled-small_20260404_120000',
        implementation='classify-api',
        dataset='labeled-small',
        timestamp='2026-04-04T12:00:00Z',
        config_snapshot_path='experiments/classify-api_labeled-small_20260404_120000/config-snapshot.json',
    )
    assert exp.implementation == 'classify-api'


def test_experiment_summary_serialization():
    summary = ExperimentSummary(
        id='classify-api_labeled-small_20260404_120000',
        implementation='classify-api',
        dataset='labeled-small',
        timestamp='2026-04-04T12:00:00Z',
        models=['claude-sonnet-4-5-20250514'],
        matched_examples=100,
        macro_accuracy=0.85,
        field_accuracy={'topic': 0.95, 'sentiment': 0.78},
        cost={'total_usd': 1.23, 'avg_usd': 0.0123},
    )
    line = json.dumps(asdict(summary))
    loaded = json.loads(line)
    assert loaded['macro_accuracy'] == 0.85


def test_field_metrics():
    metrics = FieldMetrics(field_name='topic', accuracy=0.95, total=100, correct=95)
    assert metrics.f1 == 0.0  # not computed yet, just default


def test_eval_report():
    report = EvalReport(
        experiment_id='test',
        field_metrics=[FieldMetrics(field_name='topic', accuracy=0.95, total=100, correct=95)],
        cost_total_usd=1.23,
    )
    assert len(report.field_metrics) == 1


def test_config_snapshot():
    snap = ConfigSnapshot(
        implementation='classify-api',
        platform='api',
        runner='anthropic',
        models=['claude-sonnet-4-5-20250514'],
        prompts={'system': 'You are a classifier.'},
    )
    d = asdict(snap)
    assert d['models'] == ['claude-sonnet-4-5-20250514']
    assert json.dumps(d)  # serializable
