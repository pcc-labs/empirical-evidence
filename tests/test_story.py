import json

import pytest

from autotune import story as story_mod
from autotune.story import build_story, load_routes, load_story


def test_build_story_is_ordered_by_progress():
    s = build_story("route1", target_map_id=1, routes={})
    ordinals = [b.beat_id for b in s.beats]
    progresses = [b.progress for b in s.beats]
    assert ordinals == sorted(ordinals)
    assert progresses == sorted(progresses)
    # First beat is the player's house (progress 1), last is Pewter City (progress 9).
    assert s.beats[0].map_id == 37
    assert s.beats[-1].map_id == 2


def test_target_beat_resolution():
    s = build_story("route1", target_map_id=1, routes={})
    assert s.target_beat.map_id == 1
    assert s.target_beat.name == "Viridian City"  # fallback name


def test_route_content_overrides_fallback_name():
    routes = {1: {"name": "Viridian (custom)", "waypoints": [{"x": 1, "y": 2}]}}
    s = build_story("route1", target_map_id=1, routes=routes)
    beat = s.beat_for_map(1)
    assert beat.name == "Viridian (custom)"
    assert beat.waypoints == ({"x": 1, "y": 2},)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        build_story("route1", target_map_id=999, routes={})


def test_ordinal_and_beat_lookups():
    s = build_story("route1", target_map_id=1, routes={})
    assert s.ordinal_for_map(37) == 1
    assert s.ordinal_for_map(1) == 6
    assert s.ordinal_for_map(404) is None
    assert s.beat_for_map(404) is None


def test_milestones_attached():
    s = build_story("route1", target_map_id=2, routes={})
    assert s.beat_for_map(1).milestone == "Reach Viridian City"
    assert s.beat_for_map(2).milestone == "Reach Pewter City"


def test_load_routes_skips_comment_and_bad_keys(tmp_path):
    p = tmp_path / "routes.json"
    p.write_text(json.dumps({"_comment": "x", "12": {"name": "Route 1"}, "bad": {"name": "n"}}))
    routes = load_routes(p)
    assert routes == {12: {"name": "Route 1"}}


def test_load_routes_missing_file(tmp_path):
    assert load_routes(tmp_path / "nope.json") == {}


def test_load_story_end_to_end(tmp_path):
    p = tmp_path / "routes.json"
    p.write_text(json.dumps({"1": {"name": "Viridian City", "waypoints": []}}))
    s = load_story(p, name="route1", target_map_id=1)
    assert s.name == "route1"
    assert s.target_beat.map_id == 1


def test_map_progress_matches_milestone_maps():
    # Every milestone map is a real story chapter.
    assert set(story_mod._MILESTONES).issubset(set(story_mod.MAP_PROGRESS))
