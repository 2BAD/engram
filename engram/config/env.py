"""Load environment variables from a project-local .env file."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(root: Path) -> int:
    """Load `<root>/.env` into `os.environ` without overwriting already-set variables; returns the count written."""
    env_path = root / '.env'
    if not env_path.exists():
        return 0

    count = 0
    for raw_line in env_path.read_text().splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one .env line into (key, value), or None for blanks, comments, and malformed lines."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    # Allow `export KEY=value` so users can paste from .bashrc.
    if line.startswith('export '):
        line = line[len('export ') :].lstrip()

    if '=' not in line:
        return None

    key, _, value = line.partition('=')
    key = key.strip()
    value = value.strip()

    # Strip a single pair of matching surrounding quotes (requires at least 2 chars
    # so a lone stray quote is kept verbatim rather than collapsing to empty string).
    has_matched_quotes = (
        (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))
    )
    if has_matched_quotes and len(value) > 1:
        value = value[1:-1]

    return key, value
