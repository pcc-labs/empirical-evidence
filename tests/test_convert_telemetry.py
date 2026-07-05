"""Tests for the telemetry -> SFT corpus converter."""

import json
import random
from pathlib import Path

from autotune.convert_telemetry import (
    NARRATOR_TEMPLATES,
    chat,
    damage_bucket,
    gen_battle_action,
    gen_battle_outcome,
    gen_genome,
    gen_move_choice,
    gen_narrator,
    group_battles,
    load_events,
)

FIXTURES = Path(__file__).parent / "fixtures" / "convert"


def test_load_events_parses_and_counts_skipped():
    events, skipped = load_events([FIXTURES])
    assert skipped == 1
    types = [e["event_type"] for e in events]
    assert types == [
        "battle_outcome",
        "move_result",
        "milestone",
        "battle",
        "battle",
        "battle_outcome",
        "battle",
        "battle_outcome",
        "move_result",
        "move_result",
        "map_change",
        "discovery",
        "battle_end",
    ]
    files = [e["_file"] for e in events]
    assert files == [
        "2026-06-28",
        "2026-06-28",
        "2026-06-28",
        "actions",
        "actions",
        "actions",
        "actions",
        "actions",
        "moves",
        "moves",
        "narrate",
        "narrate",
        "narrate",
    ]


def test_chat_shape():
    ex = chat("sys", "usr", "ans", "battle-outcome")
    assert ex["domain"] == "battle-outcome"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == "ans"


def test_gen_battle_outcome():
    events, _ = load_events([FIXTURES])
    examples = gen_battle_outcome(events)
    assert len(examples) == 3
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


def test_group_battles_partitions_by_outcome():
    events, _ = load_events([FIXTURES])
    groups = group_battles(events)
    won = [(turns, o) for turns, o in groups if o["won"]]
    lost = [(turns, o) for turns, o in groups if not o["won"]]
    assert len(won) == 1 and len(won[0][0]) == 2
    assert len(lost) == 1 and len(lost[0][0]) == 1


def test_gen_battle_action_only_won_battles_and_cap():
    events, _ = load_events([FIXTURES])
    examples = gen_battle_action(events, random.Random(42))
    assert len(examples) == 2  # only the 2 turns of the won battle
    assert all(e["domain"] == "battle-action" for e in examples)
    assert json.loads(examples[0]["messages"][2]["content"])["action"] == "fight"
    assert gen_battle_action(events, random.Random(42), cap=1)[0] in examples


def test_gen_genome_keeps_above_median():
    examples = gen_genome([FIXTURES / "rollouts"])
    # median battles_won = 3 -> rollout-0 (5) and rollout-2 (3) kept, rollout-1 (2) dropped
    assert len(examples) == 2
    answers = [json.loads(e["messages"][2]["content"]) for e in examples]
    assert {a["stuck_threshold"] for a in answers} == {4, 6}
    assert all(e["domain"] == "genome" for e in examples)
    assert "scen-a" in examples[0]["messages"][1]["content"]


def test_narrator_template_pools_are_deep():
    for etype in ("milestone", "map_change", "discovery", "battle_end"):
        assert len(NARRATOR_TEMPLATES[etype]) >= 5


def test_gen_narrator_deterministic():
    events, _ = load_events([FIXTURES])
    a = gen_narrator(events, random.Random(42))
    b = gen_narrator(events, random.Random(42))
    assert a == b
    # narrate.jsonl has 3 events + 2026-06-28.jsonl has 1 milestone = 4 examples
    assert len(a) == 4
    assert all(e["domain"] == "narrator" for e in a)
    assert all(e["messages"][2]["content"].strip() for e in a)
