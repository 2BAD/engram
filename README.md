# Engram

AI workflow evaluation and experimentation framework.

Teams building AI-powered features need to iterate on prompts and models, measure the impact of each change, compare alternatives across different platforms, and track what worked. Today this is done through spreadsheets, ad-hoc scripts, and platform UIs with no version history. Engram provides a structured experimentation loop: define what your workflow does, run it against labeled data, score the results, track experiments, and compare alternatives. Git is the version tracker, platforms are interchangeable, and cost is a first-class metric alongside quality.

## Install

Requires Python 3.14+.

```sh
uv add engram
```

## Quick start

`engram init` scaffolds a runnable example: a `classify` workflow (topic + sentiment), **two** implementations of the same workflow (`classify-anthropic` and `classify-openai`) so you can compare platforms immediately, and a tiny labeled `sample` dataset.

```sh
engram init                                                     # scaffold project + two implementations + sample dataset
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
engram status                                                   # verify both impls load cleanly
engram eval classify-anthropic --dataset sample                 # run against Anthropic
engram eval classify-openai --dataset sample                    # run against OpenAI
engram score <anthropic-experiment-id> --save                   # compute metrics for each
engram score <openai-experiment-id> --save
engram compare <anthropic-experiment-id> <openai-experiment-id> # accuracy, precision, recall, F1 and cost side by side
```

Rename the implementations and dataset once you replace the example with your own workflow.

## Supported runners

| Runner            | Platform                    | Notes                                                                          |
| ----------------- | --------------------------- | ------------------------------------------------------------------------------ |
| `anthropic`       | Anthropic Messages API      | Direct API calls; JSON parsed from prompt-controlled output.                   |
| `anthropic-agent` | Local Python agent          | Runs a user-supplied `entry_point` function; usage/cost returned by the agent. |
| `openai`          | OpenAI Chat Completions API | JSON mode (`response_format={"type": "json_object"}`) for reliable output.     |
| `dynamiq`         | Dynamiq hosted platform     | HTTP trigger with platform-reported cost from trace data.                      |

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
