"""Runner system for triggering workflow implementations."""

from engram.runners.base import Runner
from engram.runners.registry import get_runner

__all__ = [
    'Runner',
    'get_runner',
]
