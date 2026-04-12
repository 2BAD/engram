"""Tests for the project-local .env loader."""

import os
from pathlib import Path

import pytest

from engram.config.env import _parse_env_line, load_project_env

# --- Line parser ---


def test_parse_plain():
    assert _parse_env_line('FOO=bar') == ('FOO', 'bar')


def test_parse_double_quoted():
    assert _parse_env_line('FOO="hello world"') == ('FOO', 'hello world')


def test_parse_single_quoted():
    assert _parse_env_line("FOO='hello world'") == ('FOO', 'hello world')


def test_parse_export_prefix():
    """`.bashrc`-style export lines are accepted."""
    assert _parse_env_line('export FOO=bar') == ('FOO', 'bar')
    assert _parse_env_line('export  FOO="bar baz"') == ('FOO', 'bar baz')


def test_parse_value_containing_equals():
    """Only the first `=` separates key from value."""
    assert _parse_env_line('FOO=a=b=c') == ('FOO', 'a=b=c')


def test_parse_leading_trailing_whitespace():
    assert _parse_env_line('  FOO=bar  ') == ('FOO', 'bar')


def test_parse_blank_line():
    assert _parse_env_line('') is None
    assert _parse_env_line('   ') is None


def test_parse_comment():
    assert _parse_env_line('# this is a comment') is None
    assert _parse_env_line('  # indented comment') is None


def test_parse_malformed_line():
    """Lines without `=` are silently skipped."""
    assert _parse_env_line('NOT_A_VALID_LINE') is None


def test_parse_mismatched_quotes_kept_as_literal():
    """A leading quote with no matching trailing quote is kept verbatim."""
    assert _parse_env_line('FOO="unclosed') == ('FOO', '"unclosed')


# --- load_project_env ---


def test_load_project_env_missing_file(tmp_path: Path) -> None:
    assert load_project_env(tmp_path) == 0


def test_load_project_env_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / '.env').write_text(
        '# test env\nANTHROPIC_API_KEY=sk-ant-test\nOPENAI_API_KEY="sk-openai-test"\n\nexport OTHER=value-from-export\n'
    )
    # Make sure our tests don't leak into the real environment.
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OTHER', raising=False)

    count = load_project_env(tmp_path)

    assert count == 3
    assert os.environ['ANTHROPIC_API_KEY'] == 'sk-ant-test'
    assert os.environ['OPENAI_API_KEY'] == 'sk-openai-test'
    assert os.environ['OTHER'] == 'value-from-export'


def test_load_project_env_does_not_overwrite_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key already set in the environment wins over the .env file."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'already-set')
    (tmp_path / '.env').write_text('ANTHROPIC_API_KEY=would-overwrite\nNEW_KEY=new\n')

    count = load_project_env(tmp_path)

    assert count == 1  # only NEW_KEY got written
    assert os.environ['ANTHROPIC_API_KEY'] == 'already-set'
    assert os.environ['NEW_KEY'] == 'new'


def test_load_project_env_empty_file(tmp_path: Path) -> None:
    (tmp_path / '.env').write_text('')
    assert load_project_env(tmp_path) == 0


def test_load_project_env_only_comments_and_blanks(tmp_path: Path) -> None:
    (tmp_path / '.env').write_text('# nothing to see\n\n   \n# another\n')
    assert load_project_env(tmp_path) == 0
