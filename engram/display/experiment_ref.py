"""
Pretty-print experiment references for Rich display.

Users never need to see the long ``{impl}_{dataset}_{timestamp}`` full id in
normal flow. These formatters render the three standard shapes we use in
tables, inline messages, and resolver echoes, all driven by the same entry
dict (index row or results.json metadata — both carry ``short_id``,
``implementation``, ``dataset``, and ``timestamp``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def format_ref_short(entry: dict[str, Any]) -> str:
    """Render just ``#N`` — or a dim placeholder when short_id is missing."""
    sid = entry.get('short_id')
    return f'#{sid}' if sid is not None else '[dim]#?[/dim]'


def format_ref_medium(entry: dict[str, Any]) -> str:
    """Render ``#N impl/dataset`` for inline messages, table headers, and echoes."""
    impl = entry.get('implementation', '?')
    dataset = entry.get('dataset', '?')
    return f'{format_ref_short(entry)} {impl}/{dataset}'


def format_ref_long(entry: dict[str, Any]) -> str:
    """Render ``#N impl/dataset YYYY-MM-DD HH:MM`` for echoes and baseline show."""
    when = format_when(entry.get('timestamp', ''))
    medium = format_ref_medium(entry)
    return f'{medium} {when}' if when else medium


def format_when(timestamp_iso: str) -> str:
    """Parse an ISO timestamp and render it as ``YYYY-MM-DD HH:MM``; pass through unparseable values."""
    if not timestamp_iso:
        return ''
    try:
        return datetime.fromisoformat(timestamp_iso).strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return timestamp_iso
