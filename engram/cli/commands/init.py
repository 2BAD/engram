"""Init command: scaffold a new engram project."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()

_ENGRAM_YAML = """\
name: my-project
description: An engram evaluation project
"""

_GITIGNORE = """\
.env
__pycache__/
"""

_EXPERIMENTS_GITIGNORE = '*\n!.gitignore\n!experiments.jsonl\n!baselines.json\n'

_WORKFLOW_YAML = """\
name: classify
description: Classify support conversations by topic and sentiment
input:
  type: text
  description: A short customer support conversation
output:
  fields:
    topic:
      type: enum
      values: [billing, technical, feedback]
      description: Primary subject of the conversation
    sentiment:
      type: enum
      values: [positive, negative, neutral]
      description: Overall customer tone
scorers:
  topic: exact_match
  sentiment: exact_match
confusion_matrices:
  - topic
"""

_IMPLEMENTATION_YAML = """\
workflow: classify
platform: api
runner: anthropic
runner_config:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-sonnet-4-5-20250514
  max_tokens: "1024"
config_management:
  mode: local
"""

_SYSTEM_PROMPT = """\
You are a classifier for customer support conversations. Read the conversation
and respond with a single JSON object containing exactly two fields:

- "topic": one of "billing", "technical", or "feedback"
- "sentiment": one of "positive", "negative", or "neutral"

Return only the JSON object with no surrounding prose or markdown fences.

Example output:
{"topic": "billing", "sentiment": "negative"}
"""

_DATASET_YAML = """\
name: sample
description: Three example conversations for the quickstart
"""

_INPUT_001 = """\
Customer: I was charged twice for my subscription this month. Can you refund the duplicate charge?
Agent: I'm sorry about that, let me look into it right away.
"""

_INPUT_002 = """\
Customer: The export button on the dashboard does nothing when I click it. Is there a workaround?
Agent: Let me check if there are any known issues with the export feature.
"""

_INPUT_003 = """\
Customer: Just wanted to say the new search feature is fantastic. It saved me so much time today.
Agent: Thank you so much for letting us know, I'll pass this along to the team.
"""

_LABELS_JSON = """\
{
  "001.txt": {"topic": "billing", "sentiment": "negative"},
  "002.txt": {"topic": "technical", "sentiment": "neutral"},
  "003.txt": {"topic": "feedback", "sentiment": "positive"}
}
"""

_TEMPLATES: dict[str, str] = {
    'engram.yaml': _ENGRAM_YAML,
    '.gitignore': _GITIGNORE,
    'experiments/.gitignore': _EXPERIMENTS_GITIGNORE,
    'workflows/classify/workflow.yaml': _WORKFLOW_YAML,
    'implementations/classify-api/implementation.yaml': _IMPLEMENTATION_YAML,
    'implementations/classify-api/prompts/system.md': _SYSTEM_PROMPT,
    'datasets/sample/dataset.yaml': _DATASET_YAML,
    'datasets/sample/inputs/001.txt': _INPUT_001,
    'datasets/sample/inputs/002.txt': _INPUT_002,
    'datasets/sample/inputs/003.txt': _INPUT_003,
    'datasets/sample/labels.json': _LABELS_JSON,
}


def init_command() -> None:
    """Scaffold a new engram project in the current directory."""
    root = Path.cwd()

    if (root / 'engram.yaml').exists():
        console.print('[red]engram.yaml already exists in this directory.[/red]')
        raise typer.Exit(1)

    for rel_path, content in _TEMPLATES.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    console.print('[green]Initialized engram project with the classify example.[/green]')
    console.print()
    console.print('[bold]Next steps:[/bold]')
    console.print('  1. Set your API key: [cyan]export ANTHROPIC_API_KEY=sk-ant-...[/cyan]')
    console.print('  2. Verify the setup:  [cyan]engram status[/cyan]')
    console.print('  3. Preview cost:      [cyan]engram estimate classify-api --dataset sample[/cyan]')
    console.print('  4. Run the workflow:  [cyan]engram eval classify-api --dataset sample[/cyan]')
