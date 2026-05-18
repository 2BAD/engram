"""Tests for the analyzer's lower-level LLM call helper."""

from unittest.mock import MagicMock, patch

import pytest

from engram.analysis.analyzer import LLMCallResult, call_anthropic_messages

_FAKE_PRICING = {
    'claude-sonnet-4-5-20250514': {
        'input_cost_per_token': 0.000003,
        'output_cost_per_token': 0.000015,
    },
}


def _mock_response(text: str = 'ok', input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def test_call_anthropic_messages_returns_text_usage_and_cost():
    """Mocked response round-trips into LLMCallResult with usage and cost computed via pricing."""
    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('anthropic.Anthropic') as mock_cls,
        patch('engram.analysis.analyzer.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = _mock_response(
            text='judge says yes', input_tokens=100, output_tokens=20
        )
        result = call_anthropic_messages('claude-sonnet-4-5-20250514', 'sys', 'user')

    assert isinstance(result, LLMCallResult)
    assert result.text == 'judge says yes'
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.cost_usd == pytest.approx(100 * 3e-6 + 20 * 1.5e-5)


def test_call_anthropic_messages_forwards_max_tokens_and_temperature():
    """The judge will call this with low max_tokens and temperature=0; both must reach the API."""
    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('anthropic.Anthropic') as mock_cls,
        patch('engram.analysis.analyzer.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = _mock_response()
        call_anthropic_messages(
            'claude-sonnet-4-5-20250514',
            'sys',
            'user',
            max_tokens=256,
            temperature=0.5,
        )
        kwargs = mock_cls.return_value.messages.create.call_args.kwargs

    assert kwargs['max_tokens'] == 256
    assert kwargs['temperature'] == 0.5
    assert kwargs['system'] == 'sys'
    assert kwargs['messages'] == [{'role': 'user', 'content': 'user'}]


def test_call_anthropic_messages_default_max_tokens_and_temperature():
    """Defaults match the prior call_llm behavior so explain/suggest are unchanged."""
    with (
        patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}),
        patch('anthropic.Anthropic') as mock_cls,
        patch('engram.analysis.analyzer.load_pricing', return_value=_FAKE_PRICING),
    ):
        mock_cls.return_value.messages.create.return_value = _mock_response()
        call_anthropic_messages('claude-sonnet-4-5-20250514', 'sys', 'user')
        kwargs = mock_cls.return_value.messages.create.call_args.kwargs

    assert kwargs['max_tokens'] == 4096
    assert kwargs['temperature'] == 0.0


def test_call_anthropic_messages_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    """No API key → friendly ValueError, not the raw anthropic SDK error."""
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        call_anthropic_messages('claude-sonnet-4-5-20250514', 'sys', 'user')
