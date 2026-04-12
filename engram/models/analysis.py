"""Analysis configuration model."""

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for LLM-powered experiment analysis."""

    model: str
    max_examples: int = 30
