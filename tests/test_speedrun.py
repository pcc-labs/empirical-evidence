"""Tests for the pure path-study + observed-story derivation (Phase A).

The subprocess driver (run_speedrun) is exercised by the smoke run; here we test the parsing.
"""

from __future__ import annotations

from autotune.speedrun import build_path_study
from autotune.story import derive_story_from_run
from tests.conftest import map_change, overworld


def _battle(turn, enemy_hp, enemy_max_hp):
    return {
        "event_type": "battle",
        "turn": turn,
        "data": {
            "player_hp": 30, "player_max_hp": 30,
            "enemy_hp": enemy_hp, "enemy_max_hp": enemy_max_hp,
            "action": '{"action": "fight"}', "battle_type": 2, "map_id": 54,
        },
    }


def _battle_end(turn, won, battle_turns):
    return {
        "event_type": "battle_end",
        "turn": turn,
        "data": {
            "won": won, "battle_turns": battle_turns, "battle_type": 2, "map_id": 54,
            "opponent_species": "Geodude", "opponent_level": 12,
            "party": [{"species": "Charmander", "level": 14, "hp": 5, "max_hp": 35}],
        },
    }


# --- derive_story_from_run ---


def test_derive_story_follows_first_visit_order():
    visited = [37, 40, 0, 12, 1, 12, 0]  # backtracks at the end
    story = derive_story_from_run(visited)
    assert [b.map_id for b in story.beats] == [37, 40, 0, 12, 1]  # deduped, first-visit order
    assert story.target_beat_id == 5  # last beat by default


def test_derive_story_targets_named_map():
    story = derive_story_from_run([37, 40, 0, 12, 1, 13, 51, 2], target_map_id=2)
    assert story.target_beat.map_id == 2
    assert story.target_beat_id == 8


def test_derive_story_handles_off_canonical_maps():
    # 54 (gym interior) isn't in MAP_PROGRESS -> progress 0 but still a beat.
    story = derive_story_from_run([2, 54])
    gym = story.beat_for_map(54)
    assert gym is not None and gym.progress == 0


# --- build_path_study ---


def _full_run_events():
    maps = [37, 40, 0, 12, 1, 13, 51, 2]
    events = [overworld(37, 0)]
    turn = 1
    for prev, nxt in zip(maps, maps[1:]):
        events.append(map_change(prev, nxt, turn))
        events.append(overworld(nxt, turn))
        turn += 20
    # Brock fight on the gym map (54), then a win.
    events.append(_battle(turn, 26, 26))
    events.append(_battle(turn + 1, 0, 26))
    events.append(_battle(turn + 2, 10, 30))
    events.append(_battle_end(turn + 3, won=True, battle_turns=3))
    return events, turn + 3


def test_path_study_reaches_pewter_and_segments():
    events, end_turn = _full_run_events()
    fitness = {"turns": end_turn, "brock_won": True, "brock_turns": 3,
               "brock_lead_species": "Charmander", "brock_lead_level": 14}
    study = build_path_study(events, fitness)
    assert study.reached_pewter is True
    assert study.total_turns == end_turn
    # One segment per distinct map in order, with turns-per-segment recorded.
    assert [s.map_id for s in study.segments][:8] == [37, 40, 0, 12, 1, 13, 51, 2]
    assert all(s.turns >= 0 for s in study.segments)


def test_path_study_brock_summary():
    events, end_turn = _full_run_events()
    fitness = {"turns": end_turn, "brock_won": True, "brock_turns": 3,
               "brock_lead_species": "Charmander", "brock_lead_level": 14}
    study = build_path_study(events, fitness)
    assert study.brock.reached is True
    assert study.brock.won is True
    assert study.brock.turns == 3
    assert study.brock.lead_species == "Charmander"
    assert study.brock.gym_map_id == 54


def test_path_study_not_reached():
    # Stuck on Route 1, no Brock fight.
    events = [overworld(37, 0), map_change(37, 40, 1), map_change(40, 0, 2), map_change(0, 12, 3)]
    study = build_path_study(events, {"turns": 50, "final_map_id": 12})
    assert study.reached_pewter is False
    assert study.brock.reached is False
    assert study.brock.won is None
