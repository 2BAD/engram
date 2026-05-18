"""Init command: scaffold a new engram project."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()

_ENGRAM_YAML = """\
name: my-project
description: An engram evaluation project

analysis:
  model: claude-sonnet-4-6
"""

_GITIGNORE = """\
.env
__pycache__/

# Per-experiment artifacts are local; only the index and baselines are tracked.
experiments/*
!experiments/experiments.jsonl
!experiments/baselines.json
"""

# experiments.jsonl is append-only; union merge lets parallel appends from
# different branches coexist without conflicts. Safe because short_ids are
# computed at display time, not persisted in the index.
_GITATTRIBUTES = """\
experiments/experiments.jsonl merge=union
"""

_ENV_EXAMPLE = """\
# Copy to `.env` and fill in your keys. Engram loads `.env` from the project
# root on every command, so you don't need to export these.
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
"""

_WORKFLOW_YAML = """\
name: classify
description: Classify support conversations and extract the customer's key statement

input:
  type: text
  description: A short customer support conversation

output:
  fields:
    topic:
      type: enum
      values: [billing, technical, feedback]
      description: What the conversation is about
    sentiment:
      type: enum
      values: [positive, negative, neutral]
      description: Customer tone
    key_quote:
      type: string
      description: The customer's most important sentence, copied verbatim or lightly paraphrased

# Mix of scorers: exact_match for enums, llm_judge for the open-ended quote field
# (paraphrasing breaks string equality but is fine for this task). The judge runs at
# score time and costs a small Anthropic call per labeled example; see README.
scorers:
  topic: exact_match
  sentiment: exact_match
  key_quote:
    type: llm_judge
    criteria: |
      Does the predicted quote capture the same point as the expected quote?
      Wording can differ; small omissions are fine; meaning must match.
    model: claude-sonnet-4-6
    threshold: 0.7

confusion_matrices:
  - topic
"""

_ANTHROPIC_IMPL_YAML = """\
workflow: classify
platform: api
runner: anthropic

runner_config:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-sonnet-4-6
  max_tokens: "1024"
  # temperature 0 keeps scoring reproducible across re-runs.
  temperature: "0"
  # Cache the system prompt across runs, ~90% off on cache hits.
  # Only worth turning on once the system prompt is past ~1024 tokens.
  # prompt_cache: "true"
  # Extended thinking budget; requires temperature: "1" and max_tokens > thinking_budget.
  # thinking_budget: "4096"

config_management:
  mode: local
"""

_OPENAI_IMPL_YAML = """\
workflow: classify
platform: api
runner: openai

runner_config:
  api_key_env: OPENAI_API_KEY
  model: gpt-5.4-mini
  max_tokens: "1024"
  # temperature 0 keeps scoring reproducible across re-runs.
  temperature: "0"
  # GPT-5 / o-series reasoning controls. Forwarded only if set.
  # reasoning_effort: minimal | low | medium | high
  # verbosity: low | medium | high

config_management:
  mode: local
"""

_LITELLM_IMPL_YAML = """\
workflow: classify
platform: api
runner: litellm

# Provider comes from the model prefix: `gemini/`, `openai/`, `anthropic/`,
# `bedrock/`, `groq/`, `ollama/`, `vertex_ai/`, etc. Swap the model line to
# try another provider.
runner_config:
  api_key_env: GEMINI_API_KEY
  model: gemini/gemini-2.5-flash
  max_tokens: "1024"
  temperature: "0"
  # prompt_cache forwards to Anthropic when the model is `anthropic/...`.
  # No-op for other providers; OpenAI auto-caches above ~1024 tokens anyway.
  # prompt_cache: "true"
  # Provider-specific reasoning knobs; forwarded only when set.
  # thinking_budget for `anthropic/...`; reasoning_effort + verbosity for `openai/...`.
  # thinking_budget: "4096"
  # reasoning_effort: high
  # verbosity: low

config_management:
  mode: local
"""

_SYSTEM_PROMPT = """\
You read customer support conversations and produce a structured summary. Respond
with a single JSON object containing exactly three fields:

- "topic": one of "billing", "technical", or "feedback"
- "sentiment": one of "positive", "negative", or "neutral"
- "key_quote": the single sentence from the customer that best captures their main point

Return only the JSON object with no surrounding prose or markdown fences.

Example output:
{"topic": "billing", "sentiment": "negative", "key_quote": "I was charged twice for my subscription this month."}
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
  "001.txt": {
    "topic": "billing",
    "sentiment": "negative",
    "key_quote": "I was charged twice for my subscription this month."
  },
  "002.txt": {
    "topic": "technical",
    "sentiment": "neutral",
    "key_quote": "The export button on the dashboard does nothing when I click it."
  },
  "003.txt": {
    "topic": "feedback",
    "sentiment": "positive",
    "key_quote": "The new search feature is fantastic."
  }
}
"""

_TEMPLATES: dict[str, str] = {
    'engram.yaml': _ENGRAM_YAML,
    '.gitignore': _GITIGNORE,
    '.gitattributes': _GITATTRIBUTES,
    '.env.example': _ENV_EXAMPLE,
    'workflows/classify/workflow.yaml': _WORKFLOW_YAML,
    # Two implementations of the same workflow so users can compare platforms
    # right away. Prompts start identical; diverge them as you iterate. The
    # next-steps hint points to a third LiteLLM impl for users who install the
    # optional extra.
    'implementations/classify-anthropic/implementation.yaml': _ANTHROPIC_IMPL_YAML,
    'implementations/classify-anthropic/prompts/system.md': _SYSTEM_PROMPT,
    'implementations/classify-openai/implementation.yaml': _OPENAI_IMPL_YAML,
    'implementations/classify-openai/prompts/system.md': _SYSTEM_PROMPT,
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
    console.print('Two implementations of the same workflow, for cross-platform comparison:')
    console.print('  [bold]classify-anthropic[/bold]: Anthropic Messages API')
    console.print('  [bold]classify-openai[/bold]:    OpenAI Chat Completions API')
    console.print()
    console.print('[bold]Next steps:[/bold]')
    console.print('  1. Add at least one API key: [cyan]cp .env.example .env[/cyan] and edit it')
    console.print('  2. Check the setup:   [cyan]engram status[/cyan]')
    console.print('  3. Run the evals:')
    console.print('       [cyan]engram run classify-anthropic --dataset sample[/cyan]')
    console.print('       [cyan]engram run classify-openai --dataset sample[/cyan]')
    console.print('  4. Score each run:    [cyan]engram score <experiment-id> --save[/cyan]')
    console.print('  5. Compare platforms: [cyan]engram compare <id-a> <id-b>[/cyan]')
    console.print()
    console.print('[dim]The key_quote field is graded by an LLM judge — `engram score` will make a small[/dim]')
    console.print('[dim]Anthropic call per example. Cached on disk, so re-scoring is free.[/dim]')
    console.print()
    console.print(
        '[dim]Want LiteLLM (Gemini, Bedrock, Groq, Ollama, ...)? Install the optional extra; see README.[/dim]'
    )
