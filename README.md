# Engram

AI workflow evaluation and experimentation framework.

Teams building AI-powered features need to iterate on prompts and models, measure the impact of each change, compare alternatives across different platforms, and track what worked. Today this is done through spreadsheets, ad-hoc scripts, and platform UIs with no version history. Engram provides a structured experimentation loop: define what your workflow does, run it against labeled data, score the results, track experiments, and compare alternatives. Git is the version tracker, platforms are interchangeable, and cost is a first-class metric alongside quality.

## Install

Requires Python 3.14+.

```sh
uv add engram
```

## Quick start

```sh
engram init
engram eval <implementation> --dataset <dataset>
engram score <experiment-id> --save
engram baseline set <experiment-id>
engram compare <experiment-id> --prompts
engram baseline promote <experiment-id>
engram estimate <implementation> --dataset <dataset>
```

## Development

```sh
uv sync
uv run poe test
uv run poe coverage
uv run poe lint
uv run poe typecheck
```

## How it relates to other tools

**Langfuse** is an observability platform. It traces every LLM call in production, tracks latency and cost per user/session, and provides a dashboard for monitoring live systems. It answers: "what's happening in prod, and is it good?"

**DeepEval** is an evaluation library. It provides LLM-as-judge metrics (faithfulness, hallucination, toxicity, etc.) and integrates with pytest. It answers: "given these outputs, how good are they?"

**Engram** is an experimentation framework. It compares AI workflow implementations across platforms: sync configs, run evals against labeled datasets, score with deterministic metrics, track experiments in git, and diff what changed between any two runs. It answers: "which implementation is better, and what changed?"
