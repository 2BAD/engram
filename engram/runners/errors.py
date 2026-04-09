"""Runner exceptions surfaced at the CLI boundary."""

from __future__ import annotations


class MissingAPIKeyError(Exception):
    """Raised when a runner's required environment variable is not set. Carries the env var name for CLI display."""

    def __init__(self, env_var: str, implementation: str | None = None) -> None:
        self.env_var = env_var
        self.implementation = implementation
        message = f'Environment variable {env_var!r} is not set'
        if implementation:
            message += f' (required by implementation {implementation!r})'
        super().__init__(message)
