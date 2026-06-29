from autotune.eval_proposer import EvalReport, format_report, summarize_eval
from tests.conftest import make_verdict


def _v(reward, *, turns=400, maps=2):
    v = make_verdict(furthest_beat=int(reward), story_reward=reward)
    object.__setattr__(v, "fitness", {"turns": turns, "maps_visited": maps})
    return v


def test_passes_when_proposer_reward_higher():
    report = summarize_eval([_v(6)], [_v(5)])
    assert report.passed
    assert report.proposer_reward == 6 and report.heuristic_reward == 5


def test_tiebreak_on_turns_when_reward_equal():
    # Equal progress, proposer uses fewer turns -> pass.
    assert summarize_eval([_v(5, turns=300)], [_v(5, turns=500)]).passed
    # Equal progress, proposer uses more turns -> fail.
    assert not summarize_eval([_v(5, turns=600)], [_v(5, turns=500)]).passed


def test_fails_when_proposer_reward_lower():
    assert not summarize_eval([_v(4)], [_v(6)]).passed


def test_empty_is_a_fail():
    report = summarize_eval([], [])
    assert report.n == 0 and not report.passed


def test_averages_across_states():
    report = summarize_eval([_v(6), _v(4)], [_v(5), _v(5)])
    assert report.proposer_reward == 5.0 and report.heuristic_reward == 5.0
    assert report.n == 2


def test_format_report_shows_verdict():
    text = format_report(EvalReport(2, 6.0, 5.0, 300, 500, 3.0, 2.0, True))
    assert "PASS" in text and "proposer vs heuristic over 2" in text
    assert "FAIL" in format_report(EvalReport(1, 4.0, 6.0, 500, 400, 2.0, 3.0, False))
