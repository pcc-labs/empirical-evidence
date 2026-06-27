"""Tests for the Brock reward + verdict (autotune/brock.py).

Focus: the banded reward is strictly monotone (not-reached < lost < won; faster win > slower
win) with no flat plateau (the saturation trap from docs/experiment-findings.md), and the
telemetry extraction recovers turns/win/damage from the event stream.
"""

from __future__ import annotations

from autotune.brock import (
    BROCK_MAX_TURNS,
    brock_reward,
    extract_brock_fitness,
    verify_brock,
)
from autotune.story import build_story
from tests.conftest import map_change, overworld


def pewter_story():
    """A story targeting Pewter City (map 2, the final MAP_PROGRESS beat)."""
    return build_story("brock", target_map_id=2, routes={})


def battle(turn: int, enemy_hp: int, enemy_max_hp: int, *, battle_type: int = 2) -> dict:
    return {
        "event_type": "battle",
        "turn": turn,
        "data": {
            "player_hp": 30,
            "player_max_hp": 30,
            "enemy_hp": enemy_hp,
            "enemy_max_hp": enemy_max_hp,
            "action": '{"action": "fight"}',
            "battle_type": battle_type,
        },
    }


def battle_end(
    turn: int, won: bool, battle_turns: int, *, opponent_level: int = 12, party=None
) -> dict:
    if party is None:
        party = [{"species": "Squirtle", "level": 14, "hp": 20, "max_hp": 40}]
    return {
        "event_type": "battle_end",
        "turn": turn,
        "data": {
            "won": won,
            "battle_turns": battle_turns,
            "battle_type": 2,
            "map_id": 54,
            "opponent_species": "Geodude",
            "opponent_level": opponent_level,
            "party": party,
        },
    }


# --- reward band ordering (the core monotonicity contract) ---


def _m(**kw) -> dict:
    base = dict(
        reached_pewter=False, won=False, turns=None,
        damage_frac=0.0, whiteout=False, nav_progress=0.0,
        lead_species=None, lead_level=None,
    )
    base.update(kw)
    return base


def test_bands_are_strictly_ordered():
    not_reached = brock_reward(_m(reached_pewter=False, nav_progress=1.0))   # band max 0.9
    lost = brock_reward(_m(reached_pewter=True, won=False, damage_frac=1.0))  # band max 3.0
    won_slow = brock_reward(_m(reached_pewter=True, won=True, turns=BROCK_MAX_TURNS))
    won_fast = brock_reward(_m(reached_pewter=True, won=True, turns=1))
    # not-reached < lost < won, with the best of a lower band below the worst of the next.
    assert not_reached < 1.0 <= lost < 11.0 <= won_slow < won_fast


def test_not_reached_band_is_directional_not_flat():
    low = brock_reward(_m(nav_progress=0.2))
    high = brock_reward(_m(nav_progress=0.8))
    assert high > low  # dense gradient — no saturation while still navigating


def test_lost_band_rewards_more_damage():
    little = brock_reward(_m(reached_pewter=True, won=False, damage_frac=0.1))
    lots = brock_reward(_m(reached_pewter=True, won=False, damage_frac=0.9))
    assert lots > little  # even an all-loss generation has a selectable gradient


def test_won_band_rewards_fewer_turns():
    fast = brock_reward(_m(reached_pewter=True, won=True, turns=8))
    slow = brock_reward(_m(reached_pewter=True, won=True, turns=30))
    assert fast > slow


def test_won_turns_are_clamped():
    # A win recorded with absurd/zero turns stays inside the [11, 12) win band.
    huge = brock_reward(_m(reached_pewter=True, won=True, turns=10_000))
    zero = brock_reward(_m(reached_pewter=True, won=True, turns=0))
    assert 11.0 <= huge < 12.0
    assert 11.0 <= zero < 12.0


# --- telemetry extraction ---


def test_extract_prefers_fitness_brock_fields():
    story = pewter_story()
    events = [overworld(2, 1), battle_end(20, won=True, battle_turns=9)]
    fitness = {"brock_won": True, "brock_turns": 9, "final_map_id": 2,
               "brock_lead_species": "Squirtle", "brock_lead_level": 14}
    m = extract_brock_fitness(events, fitness, story)
    assert m["reached_pewter"] is True
    assert m["won"] is True
    assert m["turns"] == 9
    assert m["damage_frac"] == 1.0  # a win = full team cleared
    assert m["lead_species"] == "Squirtle"


def test_extract_falls_back_to_battle_end_when_fitness_missing():
    story = pewter_story()
    events = [overworld(2, 1), battle_end(20, won=False, battle_turns=14)]
    m = extract_brock_fitness(events, {}, story)
    assert m["won"] is False
    assert m["turns"] == 14


def test_extract_damage_frac_from_two_mon_fight():
    """Geodude fully fainted, Onix chipped to ~50% before a loss -> ~0.75 damage."""
    story = pewter_story()
    events = [
        battle(11, enemy_hp=26, enemy_max_hp=26),   # Geodude full
        battle(12, enemy_hp=0, enemy_max_hp=26),    # Geodude fainted
        battle(13, enemy_hp=30, enemy_max_hp=30),   # Onix sent out
        battle(14, enemy_hp=15, enemy_max_hp=30),   # Onix to 50%
        battle_end(15, won=False, battle_turns=4,
                   party=[{"species": "Squirtle", "level": 12, "hp": 0, "max_hp": 30}]),
    ]
    m = extract_brock_fitness(events, {}, story)
    assert m["won"] is False
    assert abs(m["damage_frac"] - 0.75) < 1e-6
    assert m["whiteout"] is True  # party all fainted


def test_not_reached_yields_nav_progress():
    story = pewter_story()
    # Got to Route 1 (beat 5 of 9) but no Brock fight.
    events = [overworld(37, 1), map_change(37, 40, 2), map_change(40, 0, 3), map_change(0, 12, 4)]
    m = extract_brock_fitness(events, {"final_map_id": 12}, story)
    assert m["reached_pewter"] is False
    assert 0.0 < m["nav_progress"] < 1.0


def test_verify_brock_duck_types_selection_fields():
    story = pewter_story()
    events = [overworld(2, 1), battle_end(20, won=True, battle_turns=9)]
    v = verify_brock(story, {"brock_won": True, "brock_turns": 9, "final_map_id": 2}, events)
    # selection.py reads exactly these three.
    assert v.story_reward > 11.0
    assert isinstance(v.score, float)
    assert v.on_story is True
