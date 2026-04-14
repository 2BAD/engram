"""Tests for config loading, discovery, and validation."""

from pathlib import Path

import pytest

from engram.config.discovery import discover_datasets, discover_implementations, discover_workflows, find_project_root
from engram.config.loader import load_dataset, load_implementation, load_project, load_workflow
from engram.config.validation import validate_project

WORKFLOW_YAML = """\
name: classify
description: Classify conversations
input:
  type: text
  description: A conversation transcript
output:
  fields:
    topic:
      type: enum
      values: [A, B, C]
    sentiment:
      type: enum
      values: [Positive, Negative, Neutral]
    confidence:
      type: number
scorers:
  topic: exact_match
  sentiment: exact_match
  confidence: numeric_tolerance(0.1)
confusion_matrices:
  - topic
"""

IMPL_API_YAML = """\
workflow: classify
platform: api
runner: anthropic
runner_config:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-sonnet-4-5-20250514
  max_tokens: "4096"
config_management:
  mode: local
"""

DATASET_YAML = """\
name: labeled-small
description: Small labeled test set
"""


@pytest.fixture
def full_project(tmp_path: Path) -> Path:
    """Create a project with a workflow, implementation, and dataset."""
    (tmp_path / 'engram.yaml').write_text('name: test-project\ndescription: test\n')

    wf_dir = tmp_path / 'workflows' / 'classify'
    wf_dir.mkdir(parents=True)
    (wf_dir / 'workflow.yaml').write_text(WORKFLOW_YAML)

    impl_dir = tmp_path / 'implementations' / 'classify-api'
    impl_dir.mkdir(parents=True)
    (impl_dir / 'implementation.yaml').write_text(IMPL_API_YAML)

    ds_dir = tmp_path / 'datasets' / 'labeled-small'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'dataset.yaml').write_text(DATASET_YAML)

    (tmp_path / 'experiments').mkdir()
    return tmp_path


def test_find_project_root(full_project):
    assert find_project_root(full_project) == full_project
    # From a subdirectory
    sub = full_project / 'workflows' / 'classify'
    assert find_project_root(sub) == full_project


def test_find_project_root_not_found(tmp_path):
    assert find_project_root(tmp_path) is None


def test_discover_workflows(full_project):
    assert discover_workflows(full_project) == ['classify']


def test_discover_implementations(full_project):
    assert discover_implementations(full_project) == ['classify-api']


def test_discover_datasets(full_project):
    assert discover_datasets(full_project) == ['labeled-small']


def test_discover_empty(tmp_path):
    assert discover_workflows(tmp_path) == []
    assert discover_implementations(tmp_path) == []
    assert discover_datasets(tmp_path) == []


def test_load_project(full_project):
    project = load_project(full_project)
    assert project.name == 'test-project'
    assert project.description == 'test'


def test_load_workflow(full_project):
    wf = load_workflow(full_project, 'classify')
    assert wf.name == 'classify'
    assert 'topic' in wf.output_fields
    assert wf.output_fields['topic'].values == ['A', 'B', 'C']
    assert wf.scorers['topic'] == 'exact_match'
    assert wf.confusion_matrices == ['topic']


def test_load_implementation(full_project):
    impl = load_implementation(full_project, 'classify-api')
    assert impl.workflow == 'classify'
    assert impl.platform == 'api'
    assert impl.runner == 'anthropic'
    assert impl.runner_config['model'] == 'claude-sonnet-4-5-20250514'
    assert impl.config_management.mode == 'local'


def test_load_dataset(full_project):
    ds = load_dataset(full_project, 'labeled-small')
    assert ds.name == 'labeled-small'


def test_validate_project_valid(full_project):
    errors = validate_project(full_project)
    assert errors == []


def test_validate_unknown_workflow(full_project):
    # Point implementation at a nonexistent workflow
    impl_path = full_project / 'implementations' / 'classify-api' / 'implementation.yaml'
    impl_path.write_text(IMPL_API_YAML.replace('workflow: classify', 'workflow: nonexistent'))
    errors = validate_project(full_project)
    assert any('unknown workflow' in e for e in errors)


def test_validate_scorer_unknown_field(full_project):
    # Add a scorer for a field that doesn't exist in output
    wf_path = full_project / 'workflows' / 'classify' / 'workflow.yaml'
    content = wf_path.read_text().replace(
        'scorers:\n  topic: exact_match\n',
        'scorers:\n  topic: exact_match\n  nonexistent_field: exact_match\n',
    )
    wf_path.write_text(content)
    errors = validate_project(full_project)
    assert any('unknown field' in e for e in errors)


def test_load_implementation_rejects_unknown_runner(full_project):
    impl_path = full_project / 'implementations' / 'classify-api' / 'implementation.yaml'
    impl_path.write_text(IMPL_API_YAML.replace('runner: anthropic', 'runner: bogus-runner'))

    with pytest.raises(ValueError, match='Unknown runner "bogus-runner"'):
        load_implementation(full_project, 'classify-api')


def test_load_workflow_rejects_unknown_scorer(full_project):
    wf_path = full_project / 'workflows' / 'classify' / 'workflow.yaml'
    wf_path.write_text(WORKFLOW_YAML.replace('topic: exact_match', 'topic: nonesuch_scorer'))

    with pytest.raises(ValueError, match='Unknown scorer "nonesuch_scorer"'):
        load_workflow(full_project, 'classify')


def test_load_workflow_rejects_unknown_parameterized_scorer(full_project):
    wf_path = full_project / 'workflows' / 'classify' / 'workflow.yaml'
    wf_path.write_text(WORKFLOW_YAML.replace('topic: exact_match', 'topic: bogus_match(0.5)'))

    with pytest.raises(ValueError, match='Unknown scorer "bogus_match"'):
        load_workflow(full_project, 'classify')


def test_load_workflow_accepts_custom_scorer_path(full_project):
    """Custom scorer paths (with '.') pass load-time validation without file check."""
    wf_path = full_project / 'workflows' / 'classify' / 'workflow.yaml'
    wf_path.write_text(WORKFLOW_YAML.replace('topic: exact_match', 'topic: scorers.my_custom'))

    wf = load_workflow(full_project, 'classify')
    assert wf.scorers['topic'] == 'scorers.my_custom'


def test_load_implementation_parses_transform_block(full_project):
    impl_path = full_project / 'implementations' / 'classify-api' / 'implementation.yaml'
    impl_path.write_text(
        IMPL_API_YAML + 'transform:\n  input: transforms.shape_input\n  output: transforms.shape_output\n'
    )

    impl = load_implementation(full_project, 'classify-api')
    assert impl.transform.input == 'transforms.shape_input'
    assert impl.transform.output == 'transforms.shape_output'


def test_load_implementation_transform_defaults_to_empty(full_project):
    impl = load_implementation(full_project, 'classify-api')
    assert impl.transform.input is None
    assert impl.transform.output is None


def test_load_implementation_rejects_bad_transform_name(full_project):
    impl_path = full_project / 'implementations' / 'classify-api' / 'implementation.yaml'
    impl_path.write_text(IMPL_API_YAML + 'transform:\n  input: no_dot_here\n')

    with pytest.raises(ValueError, match=r'transform\.input'):
        load_implementation(full_project, 'classify-api')
