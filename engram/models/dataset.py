"""Dataset models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Dataset metadata from dataset.yaml."""

    name: str
    description: str = ''


@dataclass
class DatasetEntry:
    """A single dataset example with input and optional labels."""

    input_file: str
    input_data: str
    labels: dict[str, Any] = field(default_factory=dict)
