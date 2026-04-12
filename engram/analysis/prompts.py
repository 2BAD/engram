"""Prompt templates for LLM-powered experiment analysis."""

from __future__ import annotations

import json
from typing import Any

EXPLAIN_SYSTEM_PROMPT = (
    'You are an expert AI evaluation analyst. Your job is to analyze experiment '
    'results from an AI workflow evaluation framework and explain why the metrics '
    'look the way they do.\n\n'
    'Your analysis should be:\n'
    '- Concrete: reference specific fields, metrics, and examples\n'
    '- Causal: explain *why* results are what they are, not just restate them\n'
    '- Actionable: suggest what the user could try to improve results\n'
    '- Concise: aim for 300-600 words; longer only if the data warrants it\n\n'
    'Format your response as markdown with clear section headers. Do not include '
    'a title or top-level heading.'
)

SUGGEST_SYSTEM_PROMPT = (
    'You are an expert AI evaluation consultant. Your job is to look at experiment '
    'results from an AI workflow evaluation framework and recommend concrete next '
    'steps to improve performance.\n\n'
    'Focus on:\n'
    '- **Priority**: which fields have the most room for improvement and the highest impact\n'
    '- **Root cause**: whether errors stem from prompt wording, model capability, '
    'output format issues, or labeling ambiguity\n'
    '- **Prompt edits**: specific, concrete changes to the prompt text (quote the relevant '
    'section and suggest a rewrite)\n'
    '- **Model choice**: whether a more capable or cheaper model would change the tradeoff\n'
    '- **Cost efficiency**: if results are good, whether a cheaper model could maintain quality\n\n'
    'Be specific and prescriptive. Do not restate metrics. Every recommendation should be '
    'something the user can act on immediately.\n\n'
    'Format your response as a numbered list of recommendations in markdown. Each item should '
    'have a short bold title and a 1-3 sentence explanation. Do not include a top-level heading.'
)

_PROMPT_TRUNCATE_CHARS = 2000


def build_single_message(context: dict[str, Any]) -> str:
    """Build the user message for single-experiment analysis."""
    parts: list[str] = []

    parts.append(f'## Experiment: {context["experiment_id"]}')
    parts.append(f'Implementation: {context["implementation"]}')
    parts.append(f'Dataset: {context["dataset"]}')
    parts.append(f'Model: {context["model"]}')
    if context.get('label'):
        parts.append(f'Label: {context["label"]}')
    parts.append('')

    if context.get('workflow_description'):
        parts.append(f'### Workflow\n{context["workflow_description"]}')
        parts.append('')

    parts.append('### Output Fields')
    for name, field in context['output_fields'].items():
        values_str = f' (values: {", ".join(field["values"])})' if field.get('values') else ''
        parts.append(f'- **{name}** ({field["type"]}{values_str}): {field["description"]}')
    parts.append('')

    for prompt_name, prompt_text in context.get('prompts', {}).items():
        parts.append(f'### Prompt: {prompt_name}')
        parts.append(prompt_text[:_PROMPT_TRUNCATE_CHARS])
        parts.append('')

    parts.append('### Metrics')
    for fm in context['field_metrics']:
        parts.append(
            f'- **{fm["field_name"]}**: accuracy={fm["accuracy"]:.1%}, '
            f'precision={fm["precision"]:.1%}, recall={fm["recall"]:.1%}, '
            f'f1={fm["f1"]:.1%} (n={fm["total"]})'
        )
    parts.append('')

    for cm in context.get('confusion_matrices', []):
        parts.append(f'### Confusion Matrix: {cm["field_name"]}')
        parts.append(_format_confusion_matrix(cm))
        parts.append('')

    cost = context['cost']
    parts.append('### Cost')
    parts.append(
        f'Total: ${cost["total"]:.4f}, Avg: ${cost["avg"]:.4f}, Median: ${cost["median"]:.4f}, P95: ${cost["p95"]:.4f}'
    )
    parts.append(
        f'Examples: {context["total_examples"]} total, {context["succeeded"]} succeeded, {context["failed"]} failed'
    )
    parts.append('')

    parts.append(f'### Example Results ({len(context["examples"])} sampled)')
    _append_examples(parts, context['examples'])

    return '\n'.join(parts)


def build_comparison_message(context: dict[str, Any]) -> str:
    """Build the user message for comparison analysis."""
    parts: list[str] = []

    a = context['experiment_a']
    b = context['experiment_b']

    parts.append('## Comparing Experiments')
    parts.append(f'**A**: {a["experiment_id"]} ({a["implementation"]}/{a["dataset"]}, model: {a["model"]})')
    parts.append(f'**B**: {b["experiment_id"]} ({b["implementation"]}/{b["dataset"]}, model: {b["model"]})')
    parts.append('')

    parts.append('### Metric Deltas (B minus A)')
    for name, delta in context['field_deltas'].items():
        regression_tag = ' [REGRESSION]' if delta['regressed'] else ''
        parts.append(
            f'- **{name}**: accuracy {delta["accuracy_delta"]:+.1%}, f1 {delta["f1_delta"]:+.1%}{regression_tag}'
        )
    parts.append('')

    parts.append('### Cost Comparison')
    parts.append(f'A: total=${context["cost_a"].get("total", 0):.4f}, avg=${context["cost_a"].get("avg", 0):.4f}')
    parts.append(f'B: total=${context["cost_b"].get("total", 0):.4f}, avg=${context["cost_b"].get("avg", 0):.4f}')
    parts.append('')

    if context.get('config_diff'):
        parts.append('### Config Changes (A -> B)')
        for line in context['config_diff']:
            parts.append(f'  {line}')
        parts.append('')

    if context.get('regressions'):
        parts.append(f'### Regressions Detected: {", ".join(context["regressions"])}')
        parts.append('')

    for label, exp in [('A', a), ('B', b)]:
        if exp.get('prompts'):
            for prompt_name, prompt_text in exp['prompts'].items():
                parts.append(f'### Prompt ({label}): {prompt_name}')
                parts.append(prompt_text[:_PROMPT_TRUNCATE_CHARS])
                parts.append('')

    for label, exp in [('A', a), ('B', b)]:
        parts.append(f'### Example Results from {label} ({len(exp["examples"])} sampled)')
        _append_examples(parts, exp['examples'])
        parts.append('')

    return '\n'.join(parts)


def _append_examples(parts: list[str], examples: list[dict[str, Any]]) -> None:
    """Append formatted example results to parts."""
    for ex in examples:
        parts.append(f'\n**{ex["input_file"]}** (status: {ex["status"]})')
        if ex.get('error'):
            parts.append(f'Error: {ex["error"]}')
        if ex.get('expected'):
            parts.append(f'Expected: {json.dumps(ex["expected"])}')
        parts.append(f'Output: {json.dumps(ex["output"])}')


def _format_confusion_matrix(cm: dict[str, Any]) -> str:
    """Render a confusion matrix as a compact markdown table."""
    matrix_labels = cm.get('labels', [])
    matrix = cm.get('matrix', {})
    if not matrix_labels:
        return '(empty)'
    header = '| |' + '|'.join(matrix_labels) + '|'
    sep = '|---|' + '|'.join(['---'] * len(matrix_labels)) + '|'
    rows = [header, sep]
    for expected in matrix_labels:
        cells = [str(matrix.get(expected, {}).get(predicted, 0)) for predicted in matrix_labels]
        rows.append(f'|{expected}|' + '|'.join(cells) + '|')
    return '\n'.join(rows)
