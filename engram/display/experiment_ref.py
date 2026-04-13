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
from pathlib import Path
from typing import Any

from rich.markup import escape


def format_ref_short(entry: dict[str, Any]) -> str:
    """Render just ``#N`` — or a dim placeholder when short_id is missing."""
    sid = entry.get('short_id')
    return f'#{sid}' if sid is not None else '[dim]#?[/dim]'


def format_ref_medium(entry: dict[str, Any]) -> str:
    """
    Render ``#N impl/dataset`` (or ``#N impl/dataset [label]``).

    Returns plain text. Callers that pass this to Rich console should wrap with
    ``rich.markup.escape()`` if the string will be embedded in markup.
    """
    impl = entry.get('implementation', '?')
    dataset = entry.get('dataset', '?')
    label = entry.get('label')
    base = f'{format_ref_short(entry)} {impl}/{dataset}'
    return f'{base} [{label}]' if label else base


def format_ref_long(entry: dict[str, Any]) -> str:
    """Render ``#N impl/dataset [label] YYYY-MM-DD HH:MM`` for echoes and baseline show."""
    when = format_when(entry.get('timestamp', ''))
    medium = format_ref_medium(entry)
    return f'{medium} {when}' if when else medium


def linkify_ref(plain_ref: str, exp_dir: Path) -> str:
    """
    Wrap a plain-text ref in an OSC 8 hyperlink pointing to the experiment directory.

    Returns Rich markup safe for ``console.print()``. The label text is escaped
    so brackets in labels don't break markup.
    """
    uri = exp_dir.as_uri()
    return f'[link={uri}]{escape(plain_ref)}[/link]'


def format_when(timestamp_iso: str) -> str:
    """Parse an ISO timestamp and render it as ``YYYY-MM-DD HH:MM``; pass through unparseable values."""
    if not timestamp_iso:
        return ''
    try:
        return datetime.fromisoformat(timestamp_iso).strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return timestamp_iso
