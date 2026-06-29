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


# --- de-flattened reward: count of beats REACHED, not the strict in-order frontier ----------
# Production telemetry never carries bag_count, so the item beats (2, 5, 7) are permanent gaps.
# The reward must still credit the catcher/sign/exit beats reached past those gaps, or the loop
# sees a flat 1 for every in-forest run and has no gradient to learn.


def test_out_of_order_beat_counts_toward_reward():
    # entered + sign read, no item/trainers: frontier is stuck at 1 (gap at beat 2), but the
    # sign (beat 6) is genuinely reached -> reward credits it.
    v = score_forest([_ow(51), _sign()])
    assert v.furthest_beat == 1  # in-order frontier, for reporting
    assert v.reward == 2.0  # enter + sign reached


def test_crossing_without_items_rewards_each_reached_beat():
    # The real forest run we care about: enter, beat both catchers, read the sign, exit — with NO
    # bag_count ever emitted. Five beats reached {1,3,4,6,8} even though the frontier stays at 1.
    events = [
        _ow(51),
        _trainer_win(),
        _trainer_win(),
        _sign(),
        {"event_type": "map_change", "turn": 5, "data": {"prev_map": 51, "new_map": 13}},
    ]
    v = score_forest(events)
    assert v.crossed is True
    assert v.furthest_beat == 1
    assert v.reward == 5.0
    assert v.per_beat == (1, 0, 1, 1, 0, 1, 0, 1)
    assert v.beats_passed == 5


# --- exit is GATED on defeating the bug catchers --------------------------------------------
# Hypothesis (the discovery engine never caught it): you cannot leave Viridian Forest without
# defeating its bug catchers. The exit beat (8) must therefore require trainer_wins >=
# REQUIRED_BUG_CATCHERS, not a bare map change. `crossed` stays the raw physical-exit fact so a
# "left without fighting" anomaly is still visible.


def _exit(turn=9):
    return {"event_type": "map_change", "turn": turn, "data": {"prev_map": 51, "new_map": 13}}


def test_exit_without_catchers_is_not_credited():
    # Map changed to Route 2 but no bug catcher was beaten -> beat 8 NOT credited, though the raw
    # physical exit is still reported on `crossed` (the anomaly the discovery engine missed).
    v = score_forest([_ow(51), _exit()])
    assert v.crossed is True  # physically left (raw fact)
    assert v.per_beat[7] == 0  # exit beat gated -> not credited
    assert v.beats_passed == 1  # only the enter beat


def test_exit_with_one_catcher_is_not_credited():
    # Only one of the two bug catchers beaten -> still short of the gate, exit not credited.
    v = score_forest([_ow(51), _trainer_win(), _exit()])
    assert v.crossed is True
    assert v.per_beat[7] == 0
    assert v.beats_passed == 2  # enter + catcher #1, no exit credit


def test_exit_after_all_catchers_is_credited():
    # Both bug catchers beaten before leaving -> exit beat credited.
    v = score_forest([_ow(51), _trainer_win(), _trainer_win(), _exit()])
    assert v.crossed is True
    assert v.per_beat[7] == 1
    # enter + catcher#1 + catcher#2 + exit = 4
    assert v.beats_passed == 4
