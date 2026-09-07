"""npc-dialogue: sentences merged from window reads, bodies from the sprite table, outcomes
measured from the same run (handout, fight, gate, blocker, stale window, talk)."""

import json
import random

from autotune.npc_corpus import (
    NPC_DOMAIN,
    OUTCOMES,
    clean_said,
    gen_npc_dialogue,
    load_truth,
)

SPRITES = {
    (26, 53, 10): {"kind": "trainer", "x": 53, "y": 10, "pic": 6},
    (26, 48, 11): {"kind": "npc", "x": 48, "y": 11, "pic": 18},
    (26, 13, 54): {"kind": "item", "x": 13, "y": 54, "pic": 61, "item": 10},
}
ITEMS = {"10": "MOON STONE", "232": "HM03"}


def _engage(ts, at, said, gained=(), run="r1", mp=26):
    return {
        "ts": f"2026-09-05T16:46:{ts:02d}+00:00",
        "run_id": run,
        "event": "supervisor.body_engaged",
        "map": mp,
        "at": list(at),
        "said": said,
        "gained": [list(g) for g in gained],
    }


def _events():
    return [
        {
            "ts": "2026-09-05T16:46:40+00:00",
            "run_id": "r1",
            "event": "supervisor.leg_start",
            "pos": [26, 30, 12],
        },
        {"ts": "2026-09-05T16:46:44+00:00", "run_id": "r1", "event": "battle.outcome", "won": True},
        _engage(45, (53, 10), "You look gentle, so I | You look gentle, so I thought"),
        _engage(50, (48, 11), "Here, take this!", gained=[(232, 1)]),
        _engage(51, (13, 54), "found a MOON STONE!", gained=[(10, 1)]),
        {
            "ts": "2026-09-05T16:46:52+00:00",
            "run_id": "r1",
            "event": "battle.outcome",
            "won": False,
        },
        _engage(53, (1, 1), "Hmph. I lost."),
        {
            "ts": "2026-09-05T16:46:54+00:00",
            "run_id": "r1",
            "event": "battle.fled",
            "pos": [26, 2, 2],
        },
        _engage(55, (2, 2), "Come back and fight me!"),
        _engage(56, (3, 3), "You can pass here only if you have the EARTHBADGE!"),
        {
            "ts": "2026-09-05T16:46:57+00:00",
            "run_id": "r1",
            "event": "supervisor.blocker_engaged",
            "body": [10, 62],
            "said": "A sleeping POKéMON blocks the way!",
        },
        {
            "ts": "2026-09-05T16:46:58+00:00",
            "run_id": "r1",
            "event": "supervisor.blocker_engaged",
            "body": [11, 62],
            "said": "Something is in the way.",
        },
        _engage(59, (5, 5), "Cell Separation System!"),
        _engage(59, (6, 5), "Cell Separation System!"),
        _engage(59, (7, 5), "Cell Separation System!"),
        _engage(59, (8, 8), " | "),
        _engage(59, (8, 9), "OPTION EXIT"),
        {
            "ts": "2026-09-05T16:46:59+00:00",
            "run_id": "r2",
            "event": "supervisor.blocker_engaged",
            "body": [1, 1],
            "said": "no map yet",
        },
        {
            "ts": "bad",
            "run_id": "r2",
            "event": "supervisor.body_engaged",
            "map": 26,
            "said": "no cell",
            "gained": [],
        },
        {"ts": "not a date", "run_id": "r2", "event": "battle.outcome", "won": True},
        _engage(59, (9, 9), "Plain talk.", run="r2"),
    ]


def test_clean_said():
    assert (
        clean_said("Yo! | Yo! Champ | Yo! Champ in making! | in making! Even I")
        == "Yo! Champ in making! Even I"
    )
    assert clean_said("A | A") == "A" and clean_said("") == ""


def test_outcomes_are_measured_from_the_run():
    rows = gen_npc_dialogue(_events(), SPRITES, ITEMS)
    by_cell = {
        (r["meta"]["map"], r["meta"]["x"], r["meta"]["y"]): json.loads(r["messages"][2]["content"])
        for r in rows
    }
    assert by_cell[(26, 53, 10)] == {
        "body": "trainer",
        "outcome": "fought-won",
        "items": [],
        "gate": None,
    }
    assert by_cell[(26, 48, 11)] == {
        "body": "npc",
        "outcome": "handed",
        "items": ["HM03"],
        "gate": None,
    }
    assert by_cell[(26, 13, 54)]["body"] == "item" and by_cell[(26, 13, 54)]["items"] == [
        "MOON STONE"
    ]
    assert (
        by_cell[(26, 1, 1)]["outcome"] == "fought-lost" and by_cell[(26, 1, 1)]["body"] == "unknown"
    )
    assert by_cell[(26, 2, 2)]["outcome"] == "fled"
    assert by_cell[(26, 3, 3)] == {
        "body": "unknown",
        "outcome": "gate",
        "items": [],
        "gate": "badge_gate",
    }
    assert by_cell[(26, 10, 62)] == {
        "body": "unknown",
        "outcome": "gate",
        "items": [],
        "gate": "sleeping_blocker",
    }
    assert by_cell[(26, 11, 62)]["outcome"] == "blocker"
    assert {by_cell[(26, x, 5)]["outcome"] for x in (5, 6, 7)} == {"stale"}
    assert by_cell[(26, 9, 9)]["outcome"] == "talk"
    assert (26, 8, 8) not in by_cell  # empty read, nothing gained: no row
    assert (26, 8, 9) not in by_cell  # the START menu read on an item-ball tile: noise, no row
    assert all(r["domain"] == NPC_DOMAIN for r in rows)
    assert all(json.loads(r["messages"][2]["content"])["outcome"] in OUTCOMES for r in rows)
    assert len(rows) == 12
    user = next(r for r in rows if r["meta"]["x"] == 53)["messages"][1]["content"]
    assert "(sprite pic 6)" in user and "'You look gentle, so I thought'" in user
    assert rows[0]["meta"]["run_id"] == "r1" and rows[0]["meta"]["occurred_at"].startswith(
        "2026-09-05"
    )


def test_deterministic():
    a = gen_npc_dialogue(_events(), SPRITES, ITEMS)
    b = gen_npc_dialogue(_events(), SPRITES, ITEMS)
    assert a == b
    random.Random(1)  # no rng involved; the generator takes none


def test_load_truth(tmp_path):
    assert load_truth(tmp_path) == ({}, {})
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "rom_truth.json").write_text(
        json.dumps(
            {
                "items": {"1": "MASTER BALL"},
                "maps": {"3": {"sprites": [{"kind": "npc", "x": 1, "y": 2, "pic": 4}]}},
            }
        )
    )
    sprites, items = load_truth(tmp_path)
    assert sprites == {(3, 1, 2): {"kind": "npc", "x": 1, "y": 2, "pic": 4}} and items == {
        "1": "MASTER BALL"
    }
