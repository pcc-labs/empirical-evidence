from autotune import verifier
from autotune.verifier import composite_score, extract_visited_maps, load_game_events, verify
from tests.conftest import map_change, overworld


def test_extract_visited_maps_orders_and_dedupes():
    events = [
        overworld(38, 1),
        overworld(38, 2),
        map_change(38, 40, 3),
        overworld(40, 4),
    ]
    assert extract_visited_maps(events) == [38, 40]


def test_verify_in_order_reaches_target(story):
    events = [
        overworld(38, 1),
        map_change(38, 40, 2),
        map_change(40, 0, 3),
        map_change(0, 12, 4),
        map_change(12, 1, 5),
    ]
    v = verify(story, {"final_map_id": 1}, events)
    assert v.furthest_beat == 6  # Viridian City
    assert v.reached_target is True
    assert v.on_story is True
    assert v.per_beat == (1, 1, 1, 1, 1, 1)
    assert v.story_reward == 6.0


def test_verify_detects_off_story_skip(story):
    # Jump straight from the bedroom (beat 2) to Viridian (beat 6), skipping 3-5.
    events = [overworld(38, 1), map_change(38, 1, 2)]
    v = verify(story, {"final_map_id": 1}, events)
    assert v.furthest_beat == 2
    assert v.on_story is False
    assert v.reached_target is False


def test_verify_partial_progress(story):
    events = [overworld(38, 1), map_change(38, 40, 2), map_change(40, 0, 3)]
    v = verify(story, {"final_map_id": 0}, events)
    assert v.furthest_beat == 4  # Pallet Town
    assert v.reached_target is False
    assert v.per_beat == (1, 1, 1, 1, 0, 0)


def test_verify_falls_back_to_fitness_without_events(story):
    v = verify(story, {"final_map_id": 12}, events=[])
    assert v.visited_maps == (12,)
    assert v.furthest_beat == 5  # Route 1


def test_verify_off_story_final_map(story):
    # A map not in the story yields no on-story progress.
    v = verify(story, {"final_map_id": 999}, events=[])
    assert v.furthest_beat == 0
    assert v.furthest_beat_name == "(start)"


def test_composite_score_progress_weight(story):
    base = composite_score({"final_map_id": 1, "turns": 100}, story)
    # Viridian (progress 6) * 1000 - turns * 0.1
    assert base == 6 * 1000 - 100 * 0.1


def test_load_game_events_reads_jsonl(tmp_path):
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "2026-06-26.jsonl").write_text(
        '{"event_type":"overworld","turn":1,"data":{"map_id":38}}\n\n'
    )
    events = load_game_events(tmp_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "overworld"


def test_load_game_events_missing_dir(tmp_path):
    assert verifier.load_game_events(tmp_path / "absent") == []
