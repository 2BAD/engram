"""Output mode detection and management for TTY vs JSON logging."""

import os
import sys
from dataclasses import dataclass


@dataclass
class OutputMode:
    """Tracks whether we're in TTY (Rich) or logging (JSON) mode."""

    use_rich: bool
    use_json_logging: bool

    @classmethod
    def detect(cls, force_json: bool = False) -> OutputMode:
        """Auto-detect output mode based on environment."""
        env_format = os.environ.get('ENGRAM_LOG_FORMAT', '').lower()
        is_tty = sys.stdout.isatty()
        use_json = force_json or env_format == 'json' or not is_tty
        return cls(use_rich=not use_json, use_json_logging=use_json)


class _ModeHolder:
    def __init__(self) -> None:
        self.mode: OutputMode | None = None


_holder = _ModeHolder()


def get_output_mode() -> OutputMode:
    """Get the current output mode, detecting if not yet set."""
    if _holder.mode is None:
        return OutputMode.detect()
    return _holder.mode


def set_output_mode(mode: OutputMode) -> None:
    """Set the global output mode explicitly."""
    _holder.mode = mode


def reset_output_mode() -> None:
    """Reset output mode to None (for testing)."""
    _holder.mode = None
