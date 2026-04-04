"""Timer utility for measuring durations."""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager


@contextmanager
def stage_timer() -> Generator[Callable[[], float]]:
    """Context manager that measures wall-clock elapsed time in milliseconds."""
    start = time.monotonic()
    result: list[float] = []

    def get_elapsed() -> float:
        return result[0] if result else (time.monotonic() - start) * 1000

    try:
        yield get_elapsed
    finally:
        result.append((time.monotonic() - start) * 1000)
