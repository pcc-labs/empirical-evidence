"""Tests for the dense forest sub-beat reward (autotune/forest_story.py).

Pure logic over synthetic telemetry: a wrong scorer can't silently hand the nudge a flat or
out-of-order reward.
"""

from __future__ import annotations

from autotune.forest_story import extract_forest_signals, score_forest


def _ow(map_id, bag=None, turn=0):
    d = {"map_id": map_id}
    if bag is not None:
        d["bag_count"] = bag
    return {"event_type": "overworld", "turn": turn, "data": d}


def _trainer_win(turn=0):
    return {"event_type": "battle_outcome", "turn": turn, "data": {"battle_type": 2, "won": True}}


def _wild_win(turn=0):
    return {"event_type": "battle_outcome", "turn": turn, "data": {"battle_type": 1, "won": True}}


def _sign(text="TRAINER TIPS: catch...", turn=0):
    return {"event_type": "discovery", "turn": turn, "data": {"text": text, "kind": "sign"}}


def test_not_in_forest_scores_zero():
    v = score_forest([_ow(12), _ow(1)])
    assert v.furthest_beat == 0 and v.reward == 0.0 and not v.crossed


def test_enter_only_is_beat_one():
    v = score_forest([_ow(51)])
    assert v.furthest_beat == 1 and v.per_beat[0] == 1 and v.per_beat[1] == 0


def test_in_order_frontier_stops_at_gap():
    # entered + sign read, but no item and no trainers -> frontier stuck at beat 1 (gap at 2).
    v = score_forest([_ow(51), _sign()])
    assert v.signals.sign_read is True
    assert v.furthest_beat == 1  # beat 6 reached out of order doesn't advance the frontier


def test_wild_win_is_not_a_trainer_beat():
    v = score_forest([_ow(51, bag=1), _wild_win(), _wild_win()])
    # beat1 enter, beat2 bag>=1; beat3 needs a TRAINER win, wild wins don't count
    assert v.furthest_beat == 2


def test_full_ladder_in_order():
    events = [
        _ow(51, bag=0),          # 1 enter
        _ow(51, bag=1),          # 2 Poke Ball
        _trainer_win(),          # 3 catcher #1
        _trainer_win(),          # 4 catcher #2
        _ow(51, bag=2),          # 5 Antidote
        _sign(),                 # 6 sign
        _ow(51, bag=3),          # 7 Potion
        {"event_type": "map_change", "turn": 9, "data": {"prev_map": 51, "new_map": 13}},  # 8 exit
    ]
    v = score_forest(events)
    assert v.furthest_beat == 8 and v.beats_passed == 8 and v.crossed
    assert v.per_beat == (1, 1, 1, 1, 1, 1, 1, 1)


def test_partial_gradient_is_monotonic_signal():
    # A run that enters, grabs the ball, beats one catcher -> reward 3 (a climbable partial).
    v = score_forest([_ow(51, bag=1), _trainer_win()])
    assert v.reward == 3.0


def test_signals_extraction_counts_trainers_only():
    s = extract_forest_signals([_trainer_win(), _wild_win(), _trainer_win()])
    assert s.trainer_wins == 2
