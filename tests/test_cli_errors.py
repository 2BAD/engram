"""Tests for friendly CLI error handling."""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
import yaml.scanner

from engram.cli.errors import run_with_error_handling


def _make_raiser(exc: BaseException):
  """Return a callable that raises *exc* when called."""
  def _raise():
    raise exc
  return _raise


class TestYAMLErrors:
  def test_malformed_yaml(self, capsys: pytest.CaptureFixture[str]):
    exc = yaml.scanner.ScannerError(
      'while parsing a flow sequence',
      yaml.error.Mark('<test>', 0, 2, 5, None, None),
      "expected ',' or ']', but got '<stream end>'",
      yaml.error.Mark('<test>', 0, 3, 0, None, None),
    )
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'Error:' in err
    assert 'invalid YAML' in err
    assert 'line 4' in err  # mark.line is 0-indexed, displayed as 1-indexed

  def test_yaml_error_without_mark(self, capsys: pytest.CaptureFixture[str]):
    exc = yaml.YAMLError('something went wrong')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'invalid YAML' in err

  def test_yaml_error_with_filename(self, capsys: pytest.CaptureFixture[str]):
    exc = yaml.scanner.ScannerError(
      'while parsing',
      yaml.error.Mark('workflow.yaml', 0, 4, 2, None, None),
      'found unexpected end',
      yaml.error.Mark('workflow.yaml', 0, 4, 10, None, None),
    )
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'workflow.yaml' in err
    assert 'line 5' in err


class TestJSONErrors:
  def test_malformed_json(self, capsys: pytest.CaptureFixture[str]):
    exc = json.JSONDecodeError('Expecting value', '{bad', 1)
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'Error:' in err
    assert 'invalid JSON' in err
    assert 'Expecting value' in err


class TestFileErrors:
  def test_file_not_found(self, capsys: pytest.CaptureFixture[str]):
    exc = FileNotFoundError('Inputs directory not found: datasets/test/inputs')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'Inputs directory not found' in err

  def test_permission_denied(self, capsys: pytest.CaptureFixture[str]):
    exc = PermissionError('[Errno 13] Permission denied: /secret/file')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'Permission denied' in err


class TestConfigErrors:
  def test_missing_key(self, capsys: pytest.CaptureFixture[str]):
    exc = KeyError('runner')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'missing required config field' in err
    assert 'runner' in err

  def test_value_error(self, capsys: pytest.CaptureFixture[str]):
    exc = ValueError('implementation "test": unknown runner "bad_runner"')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'unknown runner' in err

  def test_type_error(self, capsys: pytest.CaptureFixture[str]):
    exc = TypeError('labels.json must be a JSON object or array, got int')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'labels.json must be a JSON object or array' in err


class TestNetworkErrors:
  def test_connect_error(self, capsys: pytest.CaptureFixture[str]):
    exc = httpx.ConnectError('connection refused')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'could not connect' in err
    assert 'internet connection' in err

  def test_timeout(self, capsys: pytest.CaptureFixture[str]):
    exc = httpx.ReadTimeout('timed out')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'timed out' in err


class TestCatchAll:
  def test_unknown_exception_shows_hint(self, capsys: pytest.CaptureFixture[str]):
    exc = RuntimeError('something unexpected')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'RuntimeError' in err
    assert 'something unexpected' in err
    assert 'ENGRAM_TRACEBACK=1' in err

  def test_known_errors_hide_hint(self, capsys: pytest.CaptureFixture[str]):
    exc = FileNotFoundError('no such file')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'ENGRAM_TRACEBACK' not in err


class TestTracebackMode:
  def test_traceback_env_reraises(self, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('ENGRAM_TRACEBACK', '1')
    exc = yaml.YAMLError('test')
    with pytest.raises(yaml.YAMLError):
      run_with_error_handling(_make_raiser(exc))

  def test_traceback_zero_suppresses(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv('ENGRAM_TRACEBACK', '0')
    exc = yaml.YAMLError('test')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'invalid YAML' in err

  def test_traceback_unset_suppresses(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv('ENGRAM_TRACEBACK', raising=False)
    exc = yaml.YAMLError('test')
    with pytest.raises(SystemExit, match='1'):
      run_with_error_handling(_make_raiser(exc))

    err = capsys.readouterr().err
    assert 'invalid YAML' in err


class TestPassthrough:
  def test_system_exit_passes_through(self):
    with pytest.raises(SystemExit, match='42'):
      run_with_error_handling(_make_raiser(SystemExit(42)))

  def test_keyboard_interrupt_exits_130(self):
    with pytest.raises(SystemExit, match='130'):
      run_with_error_handling(_make_raiser(KeyboardInterrupt()))

  def test_successful_call_returns_normally(self):
    run_with_error_handling(lambda: None)
