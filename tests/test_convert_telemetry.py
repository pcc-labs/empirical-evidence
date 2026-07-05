"""Tests for the telemetry -> SFT corpus converter."""

import json
from pathlib import Path

from autotune.convert_telemetry import (
    chat,
    damage_bucket,
    gen_battle_outcome,
    gen_move_choice,
    load_events,
)

FIXTURES = Path(__file__).parent / "fixtures" / "convert"


def test_load_events_parses_and_counts_skipped():
    events, skipped = load_events([FIXTURES])
    assert skipped == 1
    types = [e["event_type"] for e in events]
    assert types == ["battle_outcome", "move_result", "milestone", "move_result", "move_result"]
    files = [e["_file"] for e in events]
    assert files == ["2026-06-28", "2026-06-28", "2026-06-28", "moves", "moves"]


def test_chat_shape():
    ex = chat("sys", "usr", "ans", "battle-outcome")
    assert ex["domain"] == "battle-outcome"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == "ans"


def test_gen_battle_outcome():
    events, _ = load_events([FIXTURES])
    examples = gen_battle_outcome(events)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["domain"] == "battle-outcome"
    user = ex["messages"][1]["content"]
    assert "Charmander (lv 6, HP 21/21)" in user
    assert "Weedle (lv 3, bug type)" in user
    assert json.loads(ex["messages"][2]["content"]) == {
        "win": True,
        "recommendation": "fight",
    }


def test_damage_bucket_boundaries():
    assert damage_bucket(0, 20, False) == "none"
    assert damage_bucket(2, 20, False) == "weak"  # 10% < 15%
    assert damage_bucket(6, 20, False) == "solid"  # 30%
    assert damage_bucket(9, 20, False) == "heavy"  # 45% > 40%
    assert damage_bucket(1, 20, True) == "heavy"  # one-shot always heavy


def test_gen_move_choice_per_row_and_best_move():
    events, _ = load_events([FIXTURES])
    examples = gen_move_choice(events)
    per_row = [e for e in examples if '"bucket"' in e["messages"][2]["content"]]
    best = [e for e in examples if '"move"' in e["messages"][2]["content"]]
    # 3 move_result rows total (1 in 2026-06-28.jsonl + 2 in moves.jsonl)
    assert len(per_row) == 3
    # exactly one matchup (Charmander vs bug) has >=2 distinct moves
    assert len(best) == 1
    assert json.loads(best[0]["messages"][2]["content"]) == {"move": "Ember"}
