# Engram

AI workflow evaluation and experimentation framework.

Teams building AI-powered features need to iterate on prompts and models, measure impact, compare alternatives across platforms, and track what worked. Today that's spreadsheets, ad-hoc scripts, and platform UIs with no version history. Engram gives you a repeatable loop: define what your workflow does, run it against labeled data, score the results, track experiments, and compare alternatives. Git tracks versions, platforms are swappable, and cost is a first-class metric alongside quality.

## Install

Install as a user-level CLI tool (recommended):

```sh
uv tool install git+https://github.com/2BAD/engram   # install; puts `engram` on $PATH
uv tool upgrade engram                               # upgrade
uv tool uninstall engram                             # remove
```

## Quick start

`engram init` scaffolds a runnable example: a `classify` workflow (topic + sentiment), **three** implementations (`classify-anthropic`, `classify-openai`, and `classify-litellm`) so you can compare platforms right away, and a tiny labeled `sample` dataset.

```sh
engram init                                         # scaffold project + three implementations + sample dataset
cp .env.example .env                                # then edit .env and paste your API keys
engram status                                       # check all impls load cleanly
engram run classify-anthropic --dataset sample      # run against Anthropic (#1)
engram run classify-openai --dataset sample         # run against OpenAI (#2)
engram run classify-litellm --dataset sample        # run against Gemini via LiteLLM (#3)
engram score --save                                 # pick a run to score interactively
engram compare                                      # pick two experiments to diff interactively
engram explain                                      # LLM-powered analysis of why metrics look the way they do
engram suggest                                      # concrete next steps to improve results
```

Each run gets a short numeric id (`#1`, `#2`, ...) that you can use in place of the full experiment identifier. You can also use `@` for the most recent run, `@~1` for the previous one, or omit the ref entirely to pick interactively. Scope with `--impl`/`--dataset` (e.g. `engram score @ --impl classify-anthropic`). Add `--label "prompt-v2"` to `engram run` to tag runs with a description.

Rename the implementations and dataset once you replace the example with your own workflow. Dataset inputs can be text files, images (JPEG, PNG, GIF, WebP), or PDFs.

## Supported runners

- `anthropic` - Anthropic Messages API
- `openai` - OpenAI Chat Completions API
- `litellm` - any provider via [LiteLLM](https://github.com/BerriAI/litellm); model prefix picks the backend (`gemini/`, `bedrock/`, `groq/`, `ollama/`, `vertex_ai/`, etc.)
- `anthropic-agent` - local Python agent
- `dynamiq` - hosted platform

Custom runners can be added by implementing the `Runner` interface.

## Prompt caching

The `anthropic` and `litellm` runners take a `prompt_cache: "true"` flag in `runner_config`. On Anthropic this cuts repeated-prompt input cost by ~90% on cache hits (every run after the first, within a 5-minute window). Don't bother for short system prompts: caching only kicks in above ~1024 tokens (~4000 chars), and the first call pays a 25% creation premium you won't get back if no reads follow.

```yaml
runner_config:
  api_key_env: ANTHROPIC_API_KEY
  model: claude-sonnet-4-6
  prompt_cache: "true"
```

OpenAI auto-caches prompts above 1024 tokens, no flag needed. `engram score` reports a `cache_hit_rate` when any caching activity shows up, and `engram compare` splits total cost into input / cache-read / cache-creation / output buckets so a regression shows up where it actually happened.

## Development

```sh
uv sync
uv run poe test
uv run poe coverage
uv run poe lint
uv run poe typecheck
```

## How it relates to other tools

**Langfuse** is an observability platform. It traces every LLM call in production, tracks latency and cost per user/session, and gives you a dashboard for monitoring live systems. "What's happening in prod, and is it good?"

**DeepEval** is an evaluation library. It ships LLM-as-judge metrics (faithfulness, hallucination, toxicity, etc.) and plugs into pytest. "Given these outputs, how good are they?"

**Engram** is an experimentation framework. It compares AI workflow implementations across platforms: sync configs, run evals against labeled datasets, score with deterministic metrics, track experiments in git, and diff what changed between any two runs. "Which implementation is better, and what changed?"
