"""Runner abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engram.models.config_snapshot import ConfigSnapshot
    from engram.models.implementation import ImplementationConfig
    from engram.models.run import RunResult


class Runner(ABC):
    """Base class for all workflow runners."""

    @abstractmethod
    def trigger(self, input_data: str, impl_config: ImplementationConfig, impl_dir: Path) -> RunResult:
        """Run the workflow with a single input and return the result."""

    @abstractmethod
    def snapshot_config(self, impl_config: ImplementationConfig, impl_dir: Path) -> ConfigSnapshot:
        """Capture the current config as a frozen snapshot."""

    def estimate_cost(
        self,
        input_data: str,
        impl_config: ImplementationConfig,
        pricing: dict[str, Any],
    ) -> float | None:
        """Estimate cost for a single input. Returns None if not supported."""
        return None
