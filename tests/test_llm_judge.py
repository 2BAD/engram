"""Tests for the LLM-as-judge scorer."""

from pathlib import Path
from unittest.mock import patch

from engram.analysis.analyzer import LLMCallResult
from engram.models.input import InputData
from engram.scoring.llm_judge import JUDGE_STATE_ATTR, JudgeState, _parse_score, llm_judge
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


def test_llm_judge_attaches_call_log_for_engine_aggregation():
    """Each judge invocation appends to a list the engine reads off the scorer attribute."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)):
        scorer = llm_judge('criteria')
        scorer('p1', 'e1')
        scorer('p2', 'e2')
        scorer('p3', 'e3')

    state = getattr(scorer, JUDGE_STATE_ATTR)
    assert isinstance(state, JudgeState)
    assert len(state.calls) == 3
    assert all(call.cost_usd == 0.0001 for call in state.calls)


def test_llm_judge_call_log_per_scorer_instance():
    """Two scorers built from the same factory call must not share their logs."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)):
        scorer_a = llm_judge('criteria a')
        scorer_b = llm_judge('criteria b')
        scorer_a('p', 'e')
        scorer_a('p', 'e')
        scorer_b('p', 'e')

    assert len(getattr(scorer_a, JUDGE_STATE_ATTR).calls) == 2
    assert len(getattr(scorer_b, JUDGE_STATE_ATTR).calls) == 1


# --- Judge response cache ---


def test_llm_judge_cache_hits_skip_the_llm_call(tmp_path: Path):
    """Same prompt twice with a cache dir set → LLM called once, second call comes from disk."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock:
        scorer = llm_judge('criteria', threshold=0.5)
        getattr(scorer, JUDGE_STATE_ATTR).cache_dir = tmp_path
        assert scorer('predicted', 'expected') is True
        assert scorer('predicted', 'expected') is True
        assert mock.call_count == 1


def test_llm_judge_cache_miss_invalidates_on_changed_input(tmp_path: Path):
    """Changing predicted/expected text rebuilds the cache key and forces a fresh call."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock:
        scorer = llm_judge('criteria', threshold=0.5)
        getattr(scorer, JUDGE_STATE_ATTR).cache_dir = tmp_path
        scorer('predicted-a', 'expected')
        scorer('predicted-b', 'expected')
        assert mock.call_count == 2


def test_llm_judge_cache_disabled_flag_bypasses_disk(tmp_path: Path):
    """state.cache_disabled forces fresh API spend even when the cache dir is set."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock:
        scorer = llm_judge('criteria', threshold=0.5)
        state = getattr(scorer, JUDGE_STATE_ATTR)
        state.cache_dir = tmp_path
        state.cache_disabled = True
        scorer('predicted', 'expected')
        scorer('predicted', 'expected')
        assert mock.call_count == 2


def test_llm_judge_writes_cache_file_on_miss(tmp_path: Path):
    """A miss persists the LLMCallResult under cache_dir as a JSON file."""
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)):
        scorer = llm_judge('criteria', threshold=0.5)
        getattr(scorer, JUDGE_STATE_ATTR).cache_dir = tmp_path
        scorer('predicted', 'expected')

    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1


def test_llm_judge_corrupt_cache_falls_back_to_call(tmp_path: Path):
    """A garbage cache file is treated as a miss; the next call refreshes it."""
    # Pre-seed cache_dir with a junk file matching no key; nothing should error on read.
    (tmp_path / 'nonsense.json').write_text('this is not json')

    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock:
        scorer = llm_judge('criteria', threshold=0.5)
        getattr(scorer, JUDGE_STATE_ATTR).cache_dir = tmp_path
        assert scorer('predicted', 'expected') is True
        assert mock.call_count == 1


def test_llm_judge_no_cache_dir_means_no_caching(tmp_path: Path):
    """Without state.cache_dir set, the scorer behaves as if no cache exists. tmp_path is just here for parity."""
    _ = tmp_path
    with patch('engram.scoring.llm_judge.call_anthropic_messages', return_value=_fake_call(0.9)) as mock:
        scorer = llm_judge('criteria', threshold=0.5)
        # state.cache_dir defaults to None.
        scorer('predicted', 'expected')
        scorer('predicted', 'expected')
        assert mock.call_count == 2
