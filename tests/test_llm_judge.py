"""Tests for the LLM-as-judge scorer."""

from unittest.mock import patch

from engram.analysis.analyzer import LLMCallResult
from engram.models.input import InputData
from engram.scoring.llm_judge import _parse_score, llm_judge
from engram.scoring.registry import resolve_scorer, scorer_accepts_input_data


def _fake_call(score: float, reason: str = 'because') -> LLMCallResult:
    return LLMCallResult(
        text=f'{{"score": {score}, "reason": "{reason}"}}',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
    )


def test_llm_judge_returns_true_when_score_at_or_above_threshold():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.85)):
        scorer = llm_judge('the predicted output is correct', threshold=0.7)
        assert scorer('predicted', 'expected') is True


def test_llm_judge_returns_false_when_score_below_threshold():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.5)):
        scorer = llm_judge('the predicted output is correct', threshold=0.7)
        assert scorer('predicted', 'expected') is False


def test_llm_judge_threshold_boundary_is_inclusive():
    """score == threshold counts as a pass (>=)."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.7)):
        scorer = llm_judge('criteria', threshold=0.7)
        assert scorer('predicted', 'expected') is True


def test_llm_judge_returns_false_on_unparseable_response():
    bad = LLMCallResult(text='not json at all', input_tokens=1, output_tokens=1, cost_usd=0.0)
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=bad):
        scorer = llm_judge('criteria')
        assert scorer('predicted', 'expected') is False


def test_llm_judge_parses_markdown_fenced_json():
    fenced = LLMCallResult(
        text='```json\n{"score": 0.9, "reason": "ok"}\n```',
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
    )
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=fenced):
        scorer = llm_judge('criteria', threshold=0.5)
        assert scorer('predicted', 'expected') is True


def test_llm_judge_forwards_model_max_tokens_and_temperature():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock_call:
        scorer = llm_judge('criteria', model='claude-haiku-4-5', max_tokens=128, threshold=0.5)
        scorer('predicted', 'expected')

        positional = mock_call.call_args.args
        keyword = mock_call.call_args.kwargs
        assert positional[0] == 'claude-haiku-4-5'
        assert keyword['max_tokens'] == 128
        assert keyword['temperature'] == 0.0


def test_llm_judge_reference_free_omits_expected_from_prompt():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock_call:
        scorer = llm_judge('criteria', threshold=0.5, reference_free=True)
        scorer('predicted out', 'ignored expected')

        user_msg = mock_call.call_args.args[2]
        assert 'predicted out' in user_msg
        assert 'ignored expected' not in user_msg
        assert 'Expected output' not in user_msg


def test_llm_judge_includes_input_data_when_provided():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock_call:
        scorer = llm_judge('criteria', threshold=0.5)
        scorer('predicted', 'expected', input_data=InputData(filename='001.txt', text='the source text'))

        user_msg = mock_call.call_args.args[2]
        assert 'Source input' in user_msg
        assert 'the source text' in user_msg


def test_llm_judge_omits_input_section_when_no_input_data():
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock_call:
        scorer = llm_judge('criteria', threshold=0.5)
        scorer('predicted', 'expected')

        user_msg = mock_call.call_args.args[2]
        assert 'Source input' not in user_msg


def test_llm_judge_scorer_declares_input_data_kwarg():
    """The engine threads input_data only to scorers that declare it. llm_judge must opt in."""
    scorer = llm_judge('criteria')
    assert scorer_accepts_input_data(scorer)


def test_resolve_scorer_dict_form_routes_to_llm_judge():
    """{type: llm_judge, criteria: ..., threshold: ...} round-trips through the registry."""
    spec = {'type': 'llm_judge', 'criteria': 'the output is correct', 'threshold': 0.6}
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.65)):
        scorer = resolve_scorer(spec)
        assert scorer('predicted', 'expected') is True


def test_parse_score_accepts_in_range_floats_and_ints():
    assert _parse_score('{"score": 0.5}') == 0.5
    assert _parse_score('{"score": 1.0}') == 1.0
    assert _parse_score('{"score": 0}') == 0.0
    assert _parse_score('{"score": 1}') == 1.0


def test_parse_score_rejects_out_of_range_or_wrong_type():
    assert _parse_score('{"score": 1.5}') is None
    assert _parse_score('{"score": -0.1}') is None
    assert _parse_score('{"score": "high"}') is None
    assert _parse_score('{"score": true}') is None
    assert _parse_score('not json') is None
    assert _parse_score('{}') is None
