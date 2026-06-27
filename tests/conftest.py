"""Shared test fixtures and synthetic-telemetry builders."""

from __future__ import annotations

import pytest

from autotune.story import build_story
from autotune.verifier import RolloutVerdict


@pytest.fixture
def story():
    """Default Route-1 story targeting Viridian City (map 1, beat 6)."""
    routes = {
        0: {"name": "Pallet Town", "waypoints": [{"x": 8, "y": 10, "note": "center"}]},
        12: {"name": "Route 1", "waypoints": []},
        1: {"name": "Viridian City", "waypoints": [{"x": 17, "y": 17, "note": "enter"}]},
    }
    return build_story("route1", target_map_id=1, routes=routes)


def overworld(map_id: int, turn: int) -> dict:
    return {
        "event_type": "overworld",
        "turn": turn,
        "data": {"map_id": map_id, "position": {"x": 1, "y": 1}},
    }


def map_change(prev_map: int, new_map: int, turn: int) -> dict:
    return {
        "event_type": "map_change",
        "turn": turn,
        "data": {"prev_map": prev_map, "new_map": new_map, "position": {"x": 1, "y": 1}},
    }


def make_verdict(
    *,
    furthest_beat: int = 0,
    beats_passed: int = 0,
    on_story: bool = True,
    reached_target: bool = False,
    story_reward: float = 0.0,
    score: float = 0.0,
) -> RolloutVerdict:
    return RolloutVerdict(
        furthest_beat=furthest_beat,
        furthest_beat_name=f"beat-{furthest_beat}",
        beats_passed=beats_passed,
        per_beat=(),
        on_story=on_story,
        reached_target=reached_target,
        story_reward=story_reward,
        score=score,
        visited_maps=(),
        fitness={},
    )
