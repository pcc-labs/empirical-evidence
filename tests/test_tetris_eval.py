from autotune.tetris_eval import gate, runs_for


def _run(arm, seed, score, pieces, tokens, late):
    return {
        "arm": arm, "seed": seed, "error": None, "policy_stats": {},
        "fitness": {"score": score, "pieces_placed": pieces,
                    "policy": {"tokens_per_decision": tokens, "late": late}},
    }


BASE = "pi/gemma4:latest/features/off+live+fixed"
TUNED = "pi/gemma4-e4b-tetris:x/features/off+live+fixed"


def _base():
    return [_run(BASE, s, 180, 30, 34.0, 0) for s in (1, 2, 3, 4, 5)]  # race 330 each


def test_gate_passes_when_all_three_rules_hold():
    tuned = [_run(TUNED, s, 380, 30, 40.0, 0) for s in (1, 2, 3, 4, 5)]  # race 530
    r = gate(tuned, _base())
    assert r.passed and r.reasons == []
    assert (r.tuned_median, r.base_median) == (530.0, 330.0)


def test_gate_fails_on_an_equal_median():
    tuned = [_run(TUNED, s, 180, 30, 34.0, 0) for s in (1, 2, 3, 4, 5)]
    r = gate(tuned, _base())
    assert not r.passed and any("median" in x for x in r.reasons)


def test_gate_uses_the_median_not_one_lucky_seed():
    tuned = [_run(TUNED, 1, 900, 30, 34.0, 0)] + [
        _run(TUNED, s, 100, 30, 34.0, 0) for s in (2, 3, 4, 5)
    ]
    assert not gate(tuned, _base()).passed


def test_gate_token_rule_is_inclusive_at_one_point_five():
    at = [_run(TUNED, s, 380, 30, 51.0, 0) for s in (1, 2, 3, 4, 5)]   # 34 * 1.5 = 51.0
    over = [_run(TUNED, s, 380, 30, 51.1, 0) for s in (1, 2, 3, 4, 5)]
    assert gate(at, _base()).passed
    r = gate(over, _base())
    assert not r.passed and any("tokens" in x for x in r.reasons)


def test_gate_fails_when_late_decisions_increase():
    tuned = [_run(TUNED, s, 380, 30, 34.0, 1) for s in (1, 2, 3, 4, 5)]
    r = gate(tuned, _base())
    assert not r.passed and any("late" in x for x in r.reasons)


def test_runs_for_selects_by_model_prefix():
    result = {"runs": _base() + [_run(TUNED, 1, 1, 1, 1.0, 0)]}
    assert len(runs_for(result, "pi/gemma4:latest")) == 5
    assert len(runs_for(result, "pi/gemma4-e4b-tetris:x")) == 1
