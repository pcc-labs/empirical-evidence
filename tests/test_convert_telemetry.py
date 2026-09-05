"""Tests for the telemetry -> SFT corpus converter."""

import json
import random
from pathlib import Path

from autotune.convert_telemetry import (
    EXIT_MEMORY_BUDGET,
    NARRATOR_TEMPLATES,
    WANTED_EVENT_TYPES,
    balance,
    chat,
    damage_bucket,
    dedupe,
    drop_unresolved,
    gen_battle_action,
    gen_battle_outcome,
    gen_genome,
    gen_move_choice,
    gen_narrator,
    group_battles,
    load_events,
    split,
)
from autotune.memlog import MemLog

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


def test_gen_battle_action_skips_lean_turns(capsys):
    full = {
        "player_species": "A",
        "player_level": 5,
        "player_hp": 9,
        "player_max_hp": 10,
        "enemy_species": "B",
        "enemy_level": 3,
        "enemy_hp": 4,
        "enemy_max_hp": 8,
        "action": '{"action": "fight", "move": "TACKLE"}',
    }
    lean = {"player_hp": 9, "player_max_hp": 10, "enemy_hp": 4, "enemy_max_hp": 8, "action": "{}"}
    events = [
        {"event_type": "battle", "turn": 1, "_file": "f", "data": lean},
        {"event_type": "battle", "turn": 2, "_file": "f", "data": full},
        {"event_type": "battle_outcome", "turn": 3, "_file": "f", "data": {"won": True}},
    ]
    out = gen_battle_action(events, random.Random(0))
    assert len(out) == 1 and "your Pokemon A (lv 5)" in out[0]["messages"][1]["content"]
    assert "skipped 1 turns" in capsys.readouterr().err


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


def _mk(domain, n):
    return [chat("s", f"u{domain}{i}", f"a{i}", domain) for i in range(n)]


def test_drop_unresolved_species_rows():
    rows = [
        chat("s", "Your Pokemon Charmander (lv 5) vs enemy #6B (lv 7).", "a", "battle-outcome"),
        chat("s", "Charmander uses EMBER against Pidgey.", "a", "move-choice"),
        chat("s", '{"enemy_species": "#A9", "result": "won"}', "a", "narrator"),
        chat("s", "hole tile 0x22 on map 12; anchor #12345 stays", "a", "handoff"),
    ]
    kept, dropped = drop_unresolved(rows)
    assert [r["domain"] for r in kept] == ["move-choice", "handoff"]
    assert dropped == {"battle-outcome": 1, "narrator": 1}


def test_dedupe_drops_exact_pairs():
    ex = _mk("battle-outcome", 3) + _mk("battle-outcome", 3)
    assert len(dedupe(ex)) == 3


def test_balance_caps_dominant_domain():
    ex = _mk("battle-action", 90) + _mk("narrator", 10)
    balanced = balance(ex, random.Random(1), max_frac=0.4)
    counts = {}
    for e in balanced:
        counts[e["domain"]] = counts.get(e["domain"], 0) + 1
    assert counts["narrator"] == 10
    total = sum(counts.values())
    assert counts["battle-action"] <= 0.4 * total + 1


def test_balance_caps_multiple_dominant_domains():
    ex = _mk("a", 50) + _mk("b", 45) + _mk("c", 5)
    balanced = balance(ex, random.Random(2), max_frac=0.4)
    counts = {}
    for e in balanced:
        counts[e["domain"]] = counts.get(e["domain"], 0) + 1
    total = sum(counts.values())
    for domain, count in counts.items():
        assert count <= 0.4 * total + 1, (domain, count, total)
    assert counts["c"] == 5

    balanced_again = balance(ex, random.Random(2), max_frac=0.4)
    assert balanced == balanced_again


def test_split_is_stratified_and_deterministic():
    ex = _mk("genome", 20) + _mk("narrator", 20)
    t1, v1 = split(ex, random.Random(7))
    t2, v2 = split(ex, random.Random(7))
    assert (t1, v1) == (t2, v2)
    assert len(v1) == 4  # 10% of each domain
    assert {e["domain"] for e in v1} == {"genome", "narrator"}


def test_end_to_end_snapshot(tmp_path):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "autotune.convert_telemetry",
        "--pk-data",
        str(FIXTURES / "game"),
        "--rollouts",
        str(FIXTURES / "rollouts"),
        "--out",
        str(tmp_path / "sft"),
        "--seed",
        "42",
        "--min-total",
        "5",
        "--pk-runs",
        str(tmp_path / "no-runs"),
    ]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    h1 = json.loads((tmp_path / "sft" / "stats.json").read_text())["corpus_sha256"]
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    h2 = json.loads((tmp_path / "sft" / "stats.json").read_text())["corpus_sha256"]
    assert r1.returncode == r2.returncode == 0
    assert h1 == h2
    train = [json.loads(x) for x in (tmp_path / "sft" / "train.jsonl").read_text().splitlines()]
    assert all("domain" not in row for row in train)
    stats = json.loads((tmp_path / "sft" / "stats.json").read_text())
    assert set(stats["vintage"]) == {"min", "max", "stamped"}  # fixtures carry no timestamps
    assert (tmp_path / "sft" / "README.md").read_text().startswith("---")


def test_load_events_drops_unread_event_types_before_parsing(tmp_path):
    lines = [
        '{"event_type":"decision","turn":1,"data":{"a":1}}',
        '{"event_type": "agent_state", "turn": 1}',
        '{"event_type":"battle_outcome","turn":2,"data":{"won":true}}',
        '{"type":"legacy","turn":3}',
        "not json at all {",
        "42",
        "[1, 2]",
    ]
    (tmp_path / "x.jsonl").write_text("\n".join(lines) + "\n")
    m = MemLog(None, stream=None)
    events, skipped = load_events([tmp_path], memlog=m)
    assert [e.get("event_type", e.get("type")) for e in events] == ["battle_outcome", "legacy"]
    assert skipped == 3  # unparseable, bare int, bare list
    assert m.kept_by_file == {str(tmp_path / "x.jsonl"): 2}
    # every generator's event type is in the allow-list, so nothing it reads is dropped
    assert {
        "battle_outcome",
        "move_result",
        "battle",
        "milestone",
        "map_change",
        "discovery",
        "battle_end",
    } <= WANTED_EVENT_TYPES


def test_since_drops_older_events_on_the_raw_line(tmp_path):
    lines = [
        '{"event_type":"battle_outcome","occurred_at":"2026-06-29T01:00:00Z","data":{}}',
        '{"event_type":"battle_outcome","occurred_at":"2026-08-15T00:00:00Z","data":{}}',
        '{"event_type":"battle_outcome","data":{}}',
    ]
    (tmp_path / "2026-08-22.jsonl").write_text("\n".join(lines) + "\n")
    events, _ = load_events([tmp_path], since="2026-08-15")
    assert [e.get("occurred_at") for e in events] == ["2026-08-15T00:00:00Z", None]
    assert len(load_events([tmp_path])[0]) == 3


def test_rows_carry_provenance():
    e = {
        "event_type": "battle_outcome",
        "_file": "2026-08-22",
        "occurred_at": "2026-08-20T10:00:00Z",
        "run_id": "r1",
        "game": "yellow",
        "data": {
            "user_species": "A",
            "user_level": 5,
            "user_hp_start": 9,
            "user_max_hp": 10,
            "user_move_types": ["Normal"],
            "enemy_species": "B",
            "enemy_level": 3,
            "enemy_type": "Bug",
            "level_gap": 2,
            "had_healing": False,
            "won": True,
        },
    }
    (row,) = gen_battle_outcome([e])
    assert row["meta"] == {
        "file": "2026-08-22",
        "occurred_at": "2026-08-20T10:00:00Z",
        "run_id": "r1",
        "game": "yellow",
    }
    assert "meta" not in chat("s", "u", "a", "d")


def test_load_events_records_one_memory_line_per_file():
    m = MemLog(None, stream=None)
    events, _ = load_events([FIXTURES / "game"], memlog=m)
    phases = [r for r in _drain(m)]
    assert phases == []  # in-process only: nothing on disk, nothing echoed
    kept = {Path(k).stem: v for k, v in m.kept_by_file.items()}
    assert kept == {"2026-06-28": 3, "actions": 5, "moves": 2, "narrate": 3}
    assert sum(kept.values()) == len(events)
    assert m.count == 1 + 4 + 1  # start + one per file + one per root


def _drain(m):
    return [] if m.path is None else m.path.read_text().splitlines()


def test_memory_budget_aborts_cleanly_with_a_trail(tmp_path):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "autotune.convert_telemetry",
        "--pk-data",
        str(FIXTURES / "game"),
        "--rollouts",
        str(FIXTURES / "rollouts"),
        "--out",
        str(tmp_path / "sft"),
        "--min-total",
        "5",
        "--pk-runs",
        str(tmp_path / "no-runs"),
        "--max-rss-gb",
        "0.001",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == EXIT_MEMORY_BUDGET, r.stderr
    assert "memory budget exceeded" in r.stderr
    assert not (tmp_path / "sft" / "corpus.jsonl").exists()
    rows = [json.loads(x) for x in (tmp_path / "sft" / "memlog.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "start" and rows[-1]["phase"] == "abort"
    assert rows[-1]["rss_gb"] > 0.001


def test_recorder_runs_load_like_any_root(tmp_path):
    run = tmp_path / "runs" / "20260905-000000-abcd"
    run.mkdir(parents=True)
    row = {
        "event_type": "battle_outcome",
        "turn": 3,
        "occurred_at": "2026-09-05T00:00:01Z",
        "run_id": "20260905-000000-abcd",
        "data": {"user_species": "Gyarados", "won": True},
    }
    (run / "events.jsonl").write_text(json.dumps(row) + "\n")
    events, skipped = load_events([tmp_path / "runs", tmp_path / "missing"])
    assert skipped == 0
    assert [e["run_id"] for e in events] == ["20260905-000000-abcd"]
    assert events[0]["_file"] == "events"


def test_memory_trail_is_written_on_success(tmp_path):
    import subprocess
    import sys

    trail = tmp_path / "elsewhere" / "trail.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "autotune.convert_telemetry",
        "--pk-data",
        str(FIXTURES / "game"),
        "--rollouts",
        str(FIXTURES / "rollouts"),
        "--out",
        str(tmp_path / "sft"),
        "--min-total",
        "5",
        "--pk-runs",
        str(tmp_path / "no-runs"),
        "--max-rss-gb",
        "0",
        "--mem-log",
        str(trail),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    phases = [json.loads(x)["phase"] for x in trail.read_text().splitlines()]
    assert phases[0] == "start" and phases[-1] == "write_corpus"
    assert phases.count("load_events.file") == 4
    for p in ("load_events", "gen_battle_outcome", "dedupe", "balance", "split"):
        assert p in phases
    assert any(line.startswith("[convert] load_events.file") for line in r.stderr.splitlines())


# ---------------------------------------------------------------------------
# Game-labelled prompts
# ---------------------------------------------------------------------------


def test_game_label_mapping():
    from autotune.convert_telemetry import game_label

    assert game_label({"game": "yellow"}) == "Yellow"
    assert game_label({"game": "red_blue"}) == "Red/Blue"
    assert game_label({}) == "Red"  # legacy telemetry predates the field
    assert game_label(None) == "Red"


def test_battle_system_names_game():
    from autotune.convert_telemetry import battle_system

    assert "Pokemon Yellow agent" in battle_system("Yellow")
    assert "Pokemon Red agent" in battle_system("Red")


def test_yellow_event_produces_yellow_prompt():
    events, _ = load_events([FIXTURES])
    outcome = next(e for e in events if e["event_type"] == "battle_outcome")
    tagged = dict(outcome)
    tagged["game"] = "yellow"
    examples = gen_battle_outcome([tagged])
    assert "Pokemon Yellow agent" in examples[0]["messages"][0]["content"]


def test_legacy_events_keep_red_prompt():
    events, _ = load_events([FIXTURES])
    examples = gen_battle_outcome(events)
    assert "Pokemon Red agent" in examples[0]["messages"][0]["content"]
