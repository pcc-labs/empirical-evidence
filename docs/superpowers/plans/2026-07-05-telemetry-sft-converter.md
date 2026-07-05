# Telemetry → SFT Converter + Retrain + Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert ~72k rows of `pokemon.game.v1` telemetry into a multi-domain SFT corpus (~2–5k examples), retrain the SmolLM3-3B LoRA on the RTX 5090, verify it beats base on held-out ground-truth rows, and publish fused weights + adapter + dataset to Hugging Face Hub.

**Architecture:** One new pure-Python module `autotune/convert_telemetry.py` (deterministic, seeded) with one generator function per domain, plus a new `autotune/eval_heldout.py` eval gate. Training/fusing reuse the existing `autotune/train_sft.py` (CUDA backend) and `autotune/package.py` unchanged. Spec is at `docs/superpowers/specs/2026-07-05-telemetry-sft-converter-design.md`.

**Tech Stack:** Python 3.12, uv, pytest, TRL/PEFT/transformers (already in project), huggingface_hub `hf` CLI.

## Global Constraints

- Always `uv run ...`, never bare `python`/`pip` (AGENTS.md).
- Tests run with `uv run python -m pytest` — plain `uv run pytest` fails to spawn in this repo.
- Lint before every commit: `uv run ruff check . && uv run ruff format --check .` (line length 120).
- Converter module lives in `autotune/` (repo convention: Python CLIs are `python -m autotune.<mod>`; `scripts/` holds shell only). This intentionally supersedes the spec's `scripts/convert_telemetry.py` path.
- Determinism: all randomness through one `random.Random(seed)`; no `datetime.now()` in corpus content.
- Output format: `{"messages": [{"role": ..., "content": ...}, ...]}`. `corpus.jsonl` and `valid.jsonl` carry an extra `"domain"` key; `train.jsonl` strips it (it feeds `SFTTrainer` directly).
- Hard failure (exit 1) if total corpus < 500 examples or any of the five generators yields 0.
- The HF upload in Task 12 is outward-facing: **confirm repo names with the user immediately before uploading.**

---

### Task 1: Event loader + fixtures

**Files:**
- Create: `autotune/convert_telemetry.py`
- Create: `tests/test_convert_telemetry.py`
- Create: `tests/fixtures/convert/game/2026-06-28.jsonl`

**Interfaces:**
- Produces: `load_events(roots: list[Path]) -> tuple[list[dict], int]` — recursively reads every `*.jsonl` under each root, returns (parsed dicts in stable order, skipped-line count). Events keep their full structure (`event_type`, `data`, `turn`, plus a `"_file"` key added by the loader naming the source file stem). Also `chat(system: str, user: str, assistant: str, domain: str) -> dict` building one example.

- [ ] **Step 1: Write fixture and failing test**

`tests/fixtures/convert/game/2026-06-28.jsonl` (note line 3 is deliberately malformed):

```
{"schema":"pokemon.game.v1","event_type":"battle_outcome","turn":34,"data":{"user_species":"Charmander","user_level":6,"user_hp_start":21,"user_max_hp":21,"user_hp_end":15,"user_move_types":["normal","fire"],"had_healing":false,"enemy_species":"Weedle","enemy_level":3,"enemy_type":"bug","level_gap":3,"battle_type":1,"turns":7,"won":true}}
{"schema":"pokemon.game.v1","event_type":"move_result","turn":31,"data":{"user_species":"Charmander","user_level":6,"move":"Ember","move_type":"fire","move_power":40,"enemy_species":"Weedle","enemy_level":3,"enemy_type":"bug","damage_dealt":6,"enemy_hp_before":16,"enemy_max_hp":16,"one_shot":false,"fainted":false}}
{this line is not json
{"schema":"pokemon.game.v1","event_type":"milestone","turn":240,"data":{"description":"Reached Viridian City!"}}
```

`tests/test_convert_telemetry.py`:

```python
"""Tests for the telemetry -> SFT corpus converter."""

import json
import random
from pathlib import Path

from autotune.convert_telemetry import chat, load_events

FIXTURES = Path(__file__).parent / "fixtures" / "convert"


def test_load_events_parses_and_counts_skipped():
    events, skipped = load_events([FIXTURES])
    assert skipped == 1
    types = [e["event_type"] for e in events]
    assert types == ["battle_outcome", "move_result", "milestone"]
    assert all(e["_file"] == "2026-06-28" for e in events)


def test_chat_shape():
    ex = chat("sys", "usr", "ans", "battle-outcome")
    assert ex["domain"] == "battle-outcome"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == "ans"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_convert_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` on `autotune.convert_telemetry`.

- [ ] **Step 3: Write minimal implementation**

`autotune/convert_telemetry.py`:

```python
"""Convert pokemon.game.v1 telemetry into a multi-domain SFT corpus.

Deterministic (seeded) converter per docs/superpowers/specs/2026-07-05-telemetry-sft-converter-design.md.
Five generators (battle-outcome, move-choice, battle-action, genome, narrator) feed one dedup +
balance + stratified-split assembly. Pure Python; unit-tested; run via
``uv run python -m autotune.convert_telemetry``.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_events(roots: list[Path]) -> tuple[list[dict], int]:
    """Read every *.jsonl under each root. Returns (events, skipped_line_count)."""
    events: list[dict] = []
    skipped = 0
    for root in roots:
        if not root.exists():
            print(f"[convert] warning: missing data root {root}")
            continue
        for path in sorted(root.rglob("*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if "event_type" in d or "type" in d:
                        d["_file"] = path.stem
                        events.append(d)
    return events, skipped


def chat(system: str, user: str, assistant: str, domain: str) -> dict:
    """Build one chat-format SFT example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "domain": domain,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_convert_telemetry.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add autotune/convert_telemetry.py tests/test_convert_telemetry.py tests/fixtures/convert/
git commit -m "feat(convert): telemetry event loader + chat example builder"
```

---

### Task 2: battle-outcome generator

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`

**Interfaces:**
- Consumes: `chat`, events from `load_events`.
- Produces: `gen_battle_outcome(events: list[dict]) -> list[dict]`; module constant `BATTLE_SYSTEM = "You are the battle advisor for a Pokemon Red agent. Answer with only the requested JSON."`

- [ ] **Step 1: Write failing test**

Append to `tests/test_convert_telemetry.py`:

```python
from autotune.convert_telemetry import gen_battle_outcome


def test_gen_battle_outcome():
    events, _ = load_events([FIXTURES])
    examples = gen_battle_outcome(events)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["domain"] == "battle-outcome"
    user = ex["messages"][1]["content"]
    assert "Charmander (lv 6, HP 21/21)" in user
    assert "Weedle (lv 3, bug type)" in user
    assert json.loads(ex["messages"][2]["content"]) == {"win": True, "recommendation": "fight"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_convert_telemetry.py::test_gen_battle_outcome -v`
Expected: FAIL (ImportError: `gen_battle_outcome`).

- [ ] **Step 3: Implement**

Add to `autotune/convert_telemetry.py`:

```python
BATTLE_SYSTEM = "You are the battle advisor for a Pokemon Red agent. Answer with only the requested JSON."


def gen_battle_outcome(events: list[dict]) -> list[dict]:
    """battle_outcome rows -> win prediction + fight/flee recommendation examples."""
    out = []
    for e in events:
        if e.get("event_type") != "battle_outcome":
            continue
        d = e["data"]
        moves = ", ".join(d["user_move_types"])
        user = (
            f"Battle start.\n"
            f"Your Pokemon: {d['user_species']} (lv {d['user_level']}, "
            f"HP {d['user_hp_start']}/{d['user_max_hp']}), move types: {moves}.\n"
            f"Enemy: {d['enemy_species']} (lv {d['enemy_level']}, {d['enemy_type']} type). "
            f"Level gap: {d['level_gap']:+d}. Healing available: {'yes' if d['had_healing'] else 'no'}.\n"
            'Will the agent win this battle, and should it fight or flee? '
            'Respond with JSON {"win": bool, "recommendation": "fight"|"flee"}.'
        )
        answer = json.dumps({"win": d["won"], "recommendation": "fight" if d["won"] else "flee"})
        out.append(chat(BATTLE_SYSTEM, user, answer, "battle-outcome"))
    return out
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `uv run python -m pytest tests/test_convert_telemetry.py -v` — all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && git add -u && git commit -m "feat(convert): battle-outcome generator"
```

---

### Task 3: move-choice generator (damage buckets + best-move)

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`
- Create: `tests/fixtures/convert/game/moves.jsonl`

**Interfaces:**
- Consumes: `chat`, `BATTLE_SYSTEM`.
- Produces: `damage_bucket(damage: int, enemy_max_hp: int, one_shot: bool) -> str` (returns `"none"|"weak"|"solid"|"heavy"`); `gen_move_choice(events: list[dict]) -> list[dict]`.

- [ ] **Step 1: Fixture + failing tests**

`tests/fixtures/convert/game/moves.jsonl` — two moves for the same (Charmander, bug) matchup so the best-move shape triggers:

```
{"schema":"pokemon.game.v1","event_type":"move_result","turn":10,"data":{"user_species":"Charmander","user_level":6,"move":"Scratch","move_type":"normal","move_power":40,"enemy_species":"Caterpie","enemy_level":4,"enemy_type":"bug","damage_dealt":3,"enemy_hp_before":18,"enemy_max_hp":18,"one_shot":false,"fainted":false}}
{"schema":"pokemon.game.v1","event_type":"move_result","turn":12,"data":{"user_species":"Charmander","user_level":6,"move":"Ember","move_type":"fire","move_power":40,"enemy_species":"Caterpie","enemy_level":4,"enemy_type":"bug","damage_dealt":9,"enemy_hp_before":15,"enemy_max_hp":18,"one_shot":false,"fainted":false}}
```

Append tests:

```python
from autotune.convert_telemetry import damage_bucket, gen_move_choice


def test_damage_bucket_boundaries():
    assert damage_bucket(0, 20, False) == "none"
    assert damage_bucket(2, 20, False) == "weak"      # 10% < 15%
    assert damage_bucket(6, 20, False) == "solid"     # 30%
    assert damage_bucket(9, 20, False) == "heavy"     # 45% > 40%
    assert damage_bucket(1, 20, True) == "heavy"      # one-shot always heavy


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_convert_telemetry.py -k move -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
def damage_bucket(damage: int, enemy_max_hp: int, one_shot: bool) -> str:
    """Bucket damage as a fraction of enemy max HP (spec: none / <15% / 15-40% / >40% or one-shot)."""
    if one_shot:
        return "heavy"
    if damage <= 0:
        return "none"
    frac = damage / max(enemy_max_hp, 1)
    if frac < 0.15:
        return "weak"
    if frac <= 0.40:
        return "solid"
    return "heavy"


def gen_move_choice(events: list[dict]) -> list[dict]:
    """move_result rows -> per-row damage buckets + aggregated best-move picks."""
    out = []
    by_matchup: dict[tuple[str, str], dict[str, list[int]]] = {}
    for e in events:
        if e.get("event_type") != "move_result":
            continue
        d = e["data"]
        bucket = damage_bucket(d["damage_dealt"], d["enemy_max_hp"], d.get("one_shot", False))
        user = (
            f"{d['user_species']} (lv {d['user_level']}) uses {d['move']} "
            f"({d['move_type']}, power {d['move_power']}) against {d['enemy_species']} "
            f"(lv {d['enemy_level']}, {d['enemy_type']} type) with "
            f"{d['enemy_hp_before']}/{d['enemy_max_hp']} HP.\n"
            "How much damage relative to the enemy's max HP? "
            'Respond with JSON {"bucket": "none"|"weak"|"solid"|"heavy"}.'
        )
        out.append(chat(BATTLE_SYSTEM, user, json.dumps({"bucket": bucket}), "move-choice"))
        key = (d["user_species"], d["enemy_type"])
        by_matchup.setdefault(key, {}).setdefault(f"{d['move']} ({d['move_type']})", []).append(
            d["damage_dealt"]
        )
    for (species, enemy_type), moves in sorted(by_matchup.items()):
        if len(moves) < 2:
            continue
        means = {m: sum(v) / len(v) for m, v in moves.items()}
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        if ranked[0][1] == ranked[1][1]:
            continue  # tie: no ground-truth winner
        moves_desc = ", ".join(sorted(means))
        best_name = ranked[0][0].split(" (")[0]
        user = (
            f"{species} is fighting a {enemy_type}-type enemy. Observed moves: {moves_desc}.\n"
            'Which move deals the most damage? Respond with JSON {"move": "..."}.'
        )
        out.append(chat(BATTLE_SYSTEM, user, json.dumps({"move": best_name}), "move-choice"))
    return out
```

- [ ] **Step 4: Run full test file, all PASS.** `uv run python -m pytest tests/test_convert_telemetry.py -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && git add -A tests autotune && git commit -m "feat(convert): move-choice generator (damage buckets + best-move)"
```

---

### Task 4: battle-action generator

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`
- Create: `tests/fixtures/convert/game/actions.jsonl`

**Interfaces:**
- Consumes: `chat`, `BATTLE_SYSTEM`.
- Produces: `group_battles(events: list[dict]) -> list[tuple[list[dict], dict]]` — per source file, `battle` events partitioned by the next `battle_outcome` at a turn >= theirs; returns (turn-events, outcome-data) pairs. `gen_battle_action(events, rng: random.Random, cap: int = 800) -> list[dict]` — examples only from won battles, seeded down-sample to `cap`.

- [ ] **Step 1: Fixture + failing test**

`tests/fixtures/convert/game/actions.jsonl` — one won battle (2 turns) and one lost battle (1 turn):

```
{"schema":"pokemon.game.v1","event_type":"battle","turn":100,"data":{"player_hp":20,"player_max_hp":21,"enemy_hp":16,"enemy_max_hp":16,"action":"{\"action\": \"fight\", \"move\": \"Ember\"}","battle_type":1,"map_id":51,"enemy_species":"Weedle","enemy_level":3,"player_species":"Charmander","player_level":6}}
{"schema":"pokemon.game.v1","event_type":"battle","turn":101,"data":{"player_hp":18,"player_max_hp":21,"enemy_hp":7,"enemy_max_hp":16,"action":"{\"action\": \"fight\", \"move\": \"Ember\"}","battle_type":1,"map_id":51,"enemy_species":"Weedle","enemy_level":3,"player_species":"Charmander","player_level":6}}
{"schema":"pokemon.game.v1","event_type":"battle_outcome","turn":102,"data":{"user_species":"Charmander","user_level":6,"user_hp_start":21,"user_max_hp":21,"user_hp_end":18,"user_move_types":["normal","fire"],"had_healing":false,"enemy_species":"Weedle","enemy_level":3,"enemy_type":"bug","level_gap":3,"battle_type":1,"turns":2,"won":true}}
{"schema":"pokemon.game.v1","event_type":"battle","turn":200,"data":{"player_hp":2,"player_max_hp":21,"enemy_hp":16,"enemy_max_hp":16,"action":"{\"action\": \"run\"}","battle_type":1,"map_id":51,"enemy_species":"Pidgey","enemy_level":9,"player_species":"Charmander","player_level":6}}
{"schema":"pokemon.game.v1","event_type":"battle_outcome","turn":201,"data":{"user_species":"Charmander","user_level":6,"user_hp_start":2,"user_max_hp":21,"user_hp_end":2,"user_move_types":["normal","fire"],"had_healing":false,"enemy_species":"Pidgey","enemy_level":9,"enemy_type":"flying","level_gap":-3,"battle_type":1,"turns":1,"won":false}}
```

Append tests:

```python
from autotune.convert_telemetry import gen_battle_action, group_battles


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
```

- [ ] **Step 2: Verify failure.** `uv run python -m pytest tests/test_convert_telemetry.py -k battle_action -v` → ImportError.

- [ ] **Step 3: Implement**

```python
import random  # add to imports at top


def group_battles(events: list[dict]) -> list[tuple[list[dict], dict]]:
    """Per source file, attach each run of `battle` turns to the next `battle_outcome`."""
    groups: list[tuple[list[dict], dict]] = []
    by_file: dict[str, list[dict]] = {}
    for e in events:
        if e.get("event_type") in ("battle", "battle_outcome"):
            by_file.setdefault(e["_file"], []).append(e)
    for _file, evs in sorted(by_file.items()):
        evs = sorted(evs, key=lambda e: e.get("turn", 0))
        pending: list[dict] = []
        for e in evs:
            if e["event_type"] == "battle":
                pending.append(e)
            else:  # battle_outcome closes the current battle
                if pending:
                    groups.append((pending, e["data"]))
                pending = []
    return groups


def gen_battle_action(events: list[dict], rng: random.Random, cap: int = 800) -> list[dict]:
    """Turns of won battles -> state -> action examples (rejection sampling on outcome)."""
    out = []
    for turns, outcome in group_battles(events):
        if not outcome.get("won"):
            continue
        for e in turns:
            d = e["data"]
            try:
                action = json.loads(d["action"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            user = (
                f"In battle: your Pokemon {d['player_species']} (lv {d['player_level']}) "
                f"HP {d['player_hp']}/{d['player_max_hp']}; enemy {d['enemy_species']} "
                f"(lv {d['enemy_level']}) HP {d['enemy_hp']}/{d['enemy_max_hp']}.\n"
                'Choose the next action. Respond with the action JSON, e.g. '
                '{"action": "fight", "move": "..."} or {"action": "run"}.'
            )
            out.append(chat(BATTLE_SYSTEM, user, json.dumps(action), "battle-action"))
    if len(out) > cap:
        out = rng.sample(out, cap)
    return out
```

- [ ] **Step 4: All tests pass.** `uv run python -m pytest tests/test_convert_telemetry.py -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && git add -A tests autotune && git commit -m "feat(convert): battle-action generator with won-battle rejection sampling"
```

---

### Task 5: genome generator

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`
- Create: `tests/fixtures/convert/rollouts/scen-a/rollout-0/genome.json` (+ fitness.json; and rollout-1, rollout-2)

**Interfaces:**
- Consumes: `chat`.
- Produces: `gen_genome(rollout_roots: list[Path]) -> list[dict]` — walks `<root>/<scenario>/rollout-*/`, ranks by (`battles_won`, `turns`), keeps rollouts with `battles_won >= median` per scenario. Module constant `GENOME_SYSTEM = "You tune a Pokemon Red agent's survival genome. Respond with only the genome JSON."`

- [ ] **Step 1: Fixtures + failing test**

Create three rollouts under `tests/fixtures/convert/rollouts/scen-a/`:

- `rollout-0/genome.json`: `{"hp_run_threshold": 0.25, "stuck_threshold": 4}` — `rollout-0/fitness.json`: `{"turns": 1500, "battles_won": 5, "maps_visited": 10}`
- `rollout-1/genome.json`: `{"hp_run_threshold": 0.4, "stuck_threshold": 8}` — `rollout-1/fitness.json`: `{"turns": 900, "battles_won": 2, "maps_visited": 3}`
- `rollout-2/genome.json`: `{"hp_run_threshold": 0.3, "stuck_threshold": 6}` — `rollout-2/fitness.json`: `{"turns": 1200, "battles_won": 3, "maps_visited": 6}`

Append test:

```python
from autotune.convert_telemetry import gen_genome


def test_gen_genome_keeps_above_median():
    examples = gen_genome([FIXTURES / "rollouts"])
    # median battles_won = 3 -> rollout-0 (5) and rollout-2 (3) kept, rollout-1 (2) dropped
    assert len(examples) == 2
    answers = [json.loads(e["messages"][2]["content"]) for e in examples]
    assert {a["stuck_threshold"] for a in answers} == {4, 6}
    assert all(e["domain"] == "genome" for e in examples)
    assert "scen-a" in examples[0]["messages"][1]["content"]
```

- [ ] **Step 2: Verify failure.** `uv run python -m pytest tests/test_convert_telemetry.py -k genome -v` → ImportError.

- [ ] **Step 3: Implement**

```python
import statistics  # add to imports

GENOME_SYSTEM = "You tune a Pokemon Red agent's survival genome. Respond with only the genome JSON."


def gen_genome(rollout_roots: list[Path]) -> list[dict]:
    """Above-median rollout genomes per scenario -> fitness-summary -> genome examples."""
    out = []
    for root in rollout_roots:
        if not root.exists():
            print(f"[convert] warning: missing rollout root {root}")
            continue
        for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rollouts = []
            for rdir in sorted(scenario_dir.glob("rollout-*")):
                gpath, fpath = rdir / "genome.json", rdir / "fitness.json"
                if not (gpath.exists() and fpath.exists()):
                    continue
                genome = json.loads(gpath.read_text())
                fitness = json.loads(fpath.read_text())
                rollouts.append((genome, fitness))
            if not rollouts:
                continue
            median_won = statistics.median(f.get("battles_won", 0) for _, f in rollouts)
            for genome, fitness in rollouts:
                if fitness.get("battles_won", 0) < median_won:
                    continue
                user = (
                    f"Scenario: {scenario_dir.name}. A rollout with this genome survived "
                    f"{fitness.get('turns', 0)} turns, won {fitness.get('battles_won', 0)} battles, "
                    f"and visited {fitness.get('maps_visited', 0)} maps.\n"
                    "Propose the genome JSON that achieved this."
                )
                out.append(chat(GENOME_SYSTEM, user, json.dumps(genome, sort_keys=True), "genome"))
    return out
```

- [ ] **Step 4: All tests pass.** `uv run python -m pytest tests/test_convert_telemetry.py -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && git add -A tests autotune && git commit -m "feat(convert): genome generator from above-median rollouts"
```

---

### Task 6: narrator generator

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`
- Create: `tests/fixtures/convert/game/narrate.jsonl`

**Interfaces:**
- Consumes: `chat`.
- Produces: `gen_narrator(events: list[dict], rng: random.Random) -> list[dict]`; `NARRATOR_TEMPLATES: dict[str, list[str]]` with ≥5 templates for each of `milestone`, `map_change`, `discovery`, `battle_end`; `NARRATOR_SYSTEM = "You are the live commentator for an autonomous Pokemon Red run. Reply with one short sentence."`

- [ ] **Step 1: Fixture + failing test**

`tests/fixtures/convert/game/narrate.jsonl`:

```
{"schema":"pokemon.game.v1","event_type":"map_change","turn":50,"data":{"from_map":"Route 1","to_map":"Viridian City"}}
{"schema":"pokemon.game.v1","event_type":"discovery","turn":60,"data":{"map_id":51,"position":{"x":17,"y":47},"kind":"dialogue","text":"Got away safely!"}}
{"schema":"pokemon.game.v1","event_type":"battle_end","turn":70,"data":{"result":"won","enemy_species":"Weedle"}}
```

Append tests:

```python
from autotune.convert_telemetry import NARRATOR_TEMPLATES, gen_narrator


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
```

- [ ] **Step 2: Verify failure.** `uv run python -m pytest tests/test_convert_telemetry.py -k narrator -v` → ImportError.

- [ ] **Step 3: Implement**

```python
NARRATOR_SYSTEM = "You are the live commentator for an autonomous Pokemon Red run. Reply with one short sentence."

NARRATOR_TEMPLATES: dict[str, list[str]] = {
    "milestone": [
        "Huge moment: {description}",
        "The run just hit a milestone — {description}",
        "Checkpoint reached: {description}",
        "That's the milestone the chat was waiting for: {description}",
        "Progress locked in: {description}",
    ],
    "map_change": [
        "The agent crosses from {from_map} into {to_map}.",
        "New area: leaving {from_map}, entering {to_map}.",
        "Transition — {from_map} is behind us, {to_map} ahead.",
        "The party steps out of {from_map} and into {to_map}.",
        "Map change: {from_map} to {to_map}.",
    ],
    "discovery": [
        "Found something: {text}",
        "The agent uncovers a clue — {text}",
        "On-screen text spotted: {text}",
        "A discovery in the overworld: {text}",
        "New info just dropped: {text}",
    ],
    "battle_end": [
        "Battle over — the agent {result} against {enemy_species}.",
        "That fight with {enemy_species} ends: {result}.",
        "Result vs {enemy_species}: {result}.",
        "The {enemy_species} encounter wraps up — {result}.",
        "Dust settles on the {enemy_species} battle: {result}.",
    ],
}


def gen_narrator(events: list[dict], rng: random.Random) -> list[dict]:
    """Notable events -> one-sentence play-by-play from seeded template pools."""
    out = []
    for e in events:
        etype = e.get("event_type")
        if etype not in NARRATOR_TEMPLATES:
            continue
        d = dict(e.get("data", {}))
        d.setdefault("description", "")
        d.setdefault("from_map", "the last area")
        d.setdefault("to_map", f"map {d.get('map_id', '?')}")
        d.setdefault("text", d.get("kind", ""))
        d.setdefault("result", "is decided")
        d.setdefault("enemy_species", "the enemy")
        template = rng.choice(NARRATOR_TEMPLATES[etype])
        try:
            sentence = template.format(**d)
        except (KeyError, IndexError):
            continue
        user = f"Narrate this game event for the stream overlay in one sentence:\n{json.dumps(e.get('data', {}), sort_keys=True)}"
        out.append(chat(NARRATOR_SYSTEM, user, sentence, "narrator"))
    return out
```

- [ ] **Step 4: All tests pass.** `uv run python -m pytest tests/test_convert_telemetry.py -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && git add -A tests autotune && git commit -m "feat(convert): narrator generator with seeded template pools"
```

---

### Task 7: assembly — dedup, balance, split, stats, CLI

**Files:**
- Modify: `autotune/convert_telemetry.py`
- Modify: `tests/test_convert_telemetry.py`

**Interfaces:**
- Consumes: all five generators.
- Produces:
  - `dedupe(examples: list[dict]) -> list[dict]` (SHA-256 of user+assistant content, first-seen wins)
  - `balance(examples: list[dict], rng: random.Random, max_frac: float = 0.4) -> list[dict]`
  - `split(examples: list[dict], rng: random.Random, valid_frac: float = 0.1) -> tuple[list, list]` (stratified by domain)
  - `write_corpus(out_dir: Path, examples, train, valid, skipped: int, seed: int) -> dict` (writes 4 files, returns stats)
  - `main(argv=None) -> int` — CLI: `--pk-data`, `--ee-data`, `--rollouts` (repeatable), `--out`, `--seed`; exit 1 on thresholds.

- [ ] **Step 1: Failing tests**

```python
from autotune.convert_telemetry import balance, dedupe, split


def _mk(domain, n):
    return [chat("s", f"u{domain}{i}", f"a{i}", domain) for i in range(n)]


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


def test_split_is_stratified_and_deterministic():
    ex = _mk("genome", 20) + _mk("narrator", 20)
    t1, v1 = split(ex, random.Random(7))
    t2, v2 = split(ex, random.Random(7))
    assert (t1, v1) == (t2, v2)
    assert len(v1) == 4  # 10% of each domain
    assert {e["domain"] for e in v1} == {"genome", "narrator"}


def test_end_to_end_snapshot(tmp_path):
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "autotune.convert_telemetry",
        "--pk-data", str(FIXTURES / "game"),
        "--rollouts", str(FIXTURES / "rollouts"),
        "--out", str(tmp_path / "sft"),
        "--seed", "42", "--min-total", "5",
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
```

(Note: the CLI gets a `--min-total` flag, default 500, so fixtures can exercise the pipeline end-to-end.)

- [ ] **Step 2: Verify failure.** `uv run python -m pytest tests/test_convert_telemetry.py -k "dedupe or balance or split or snapshot" -v` → ImportError.

- [ ] **Step 3: Implement**

```python
import argparse   # add to imports
import hashlib
import sys

DOMAINS = ("battle-outcome", "move-choice", "battle-action", "genome", "narrator")


def _key(ex: dict) -> str:
    payload = ex["messages"][1]["content"] + "\x00" + ex["messages"][2]["content"]
    return hashlib.sha256(payload.encode()).hexdigest()


def dedupe(examples: list[dict]) -> list[dict]:
    """Drop exact (user, assistant) duplicates, first-seen wins."""
    seen, out = set(), []
    for ex in examples:
        k = _key(ex)
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return out


def balance(examples: list[dict], rng: random.Random, max_frac: float = 0.4) -> list[dict]:
    """Down-sample any domain above max_frac of the corpus (seeded)."""
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    changed = True
    while changed:
        changed = False
        total = sum(len(v) for v in by_domain.values())
        for domain, rows in by_domain.items():
            cap = int(max_frac * total)
            if len(rows) > cap and len(by_domain) > 1:
                by_domain[domain] = rng.sample(rows, cap)
                changed = True
                break
    out = [ex for d in sorted(by_domain) for ex in by_domain[d]]
    return out


def split(examples: list[dict], rng: random.Random, valid_frac: float = 0.1) -> tuple[list, list]:
    """Stratified train/valid split, deterministic under a fixed rng."""
    train, valid = [], []
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    for domain in sorted(by_domain):
        rows = by_domain[domain][:]
        rng.shuffle(rows)
        n_valid = max(1, int(len(rows) * valid_frac)) if len(rows) >= 2 else 0
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    return train, valid


def write_corpus(out_dir: Path, examples, train, valid, skipped: int, seed: int) -> dict:
    """Write corpus/train/valid/stats files; returns the stats dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_text = "\n".join(json.dumps(ex, sort_keys=True) for ex in examples) + "\n"
    (out_dir / "corpus.jsonl").write_text(corpus_text)
    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps({"messages": ex["messages"]}, sort_keys=True) + "\n")
    with open(out_dir / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex["domain"]] = counts.get(ex["domain"], 0) + 1
    stats = {
        "total": len(examples),
        "train": len(train),
        "valid": len(valid),
        "domains": counts,
        "skipped_lines": skipped,
        "seed": seed,
        "corpus_sha256": hashlib.sha256(corpus_text.encode()).hexdigest(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert pokemon.game.v1 telemetry into an SFT corpus.")
    p.add_argument("--pk-data", type=Path, default=Path("../pokemon-kafka/data"))
    p.add_argument("--ee-data", type=Path, default=Path("data/telemetry"))
    p.add_argument("--rollouts", type=Path, action="append", default=None,
                   help="rollout roots (repeatable); default: out/rollouts out/harvest")
    p.add_argument("--out", type=Path, default=Path("data/sft_v3"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-total", type=int, default=500)
    p.add_argument("--action-cap", type=int, default=800)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    events, skipped = load_events([args.pk_data, args.ee_data])
    rollout_roots = args.rollouts or [Path("out/rollouts"), Path("out/harvest")]

    examples = (
        gen_battle_outcome(events)
        + gen_move_choice(events)
        + gen_battle_action(events, rng, cap=args.action_cap)
        + gen_genome(rollout_roots)
        + gen_narrator(events, rng)
    )
    examples = dedupe(examples)
    examples = balance(examples, rng)
    train, valid = split(examples, rng)
    stats = write_corpus(args.out, examples, train, valid, skipped, args.seed)
    print(json.dumps(stats, indent=2, sort_keys=True))

    empty = [d for d in DOMAINS if stats["domains"].get(d, 0) == 0]
    if empty:
        print(f"[convert] FATAL: empty domains: {empty}", file=sys.stderr)
        return 1
    if stats["total"] < args.min_total:
        print(f"[convert] FATAL: only {stats['total']} examples (< {args.min_total})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: All tests pass.** `uv run python -m pytest tests/test_convert_telemetry.py -v`

Note: `test_end_to_end_snapshot` needs every domain non-empty from fixtures — Tasks 1–6 fixtures already cover all five (battle_outcome ✓, move_result ✓, battle+outcome ✓, rollouts ✓, milestone/map_change/discovery/battle_end ✓). If the CLI exits 1 on empty domains from fixtures, fix the fixture, not the threshold.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check . && git add -A tests autotune && git commit -m "feat(convert): dedup, balance, stratified split, stats, CLI"
```

---

### Task 8: run the converter on real data

**Files:**
- Create (generated): `data/sft_v3/{corpus,train,valid}.jsonl`, `data/sft_v3/stats.json`

**Interfaces:**
- Consumes: the full CLI from Task 7.
- Produces: the real corpus that Task 9 trains on.

- [ ] **Step 1: Run the converter**

```bash
cd /home/bdougie/code/pcc-labs/empirical-evidence
uv run python -m autotune.convert_telemetry \
  --pk-data ../pokemon-kafka/data \
  --ee-data data/telemetry \
  --out data/sft_v3 --seed 42
```

Expected: exit 0; stats printed with total in the 2,000–5,000 range; all five domains non-zero; `skipped_lines` small relative to input.

- [ ] **Step 2: Sanity-read 10 examples**

```bash
shuf -n 10 --random-source=<(yes) data/sft_v3/corpus.jsonl | uv run python -c "
import sys, json
for line in sys.stdin:
    ex = json.loads(line)
    print('---', ex['domain']); print(ex['messages'][1]['content'][:200]); print('=>', ex['messages'][2]['content'][:120])
"
```

Check: prompts read sensibly, answers are valid JSON where expected, no empty strings, no `#NN`-species-only garbage dominating battle-action. If a systemic problem appears, fix the generator (with a regression test) before proceeding.

- [ ] **Step 3: Commit the corpus**

```bash
git add data/sft_v3/ && git commit -m "data: sft_v3 multi-domain corpus from telemetry (seed 42)"
```

---

### Task 9: retrain LoRA on the 5090

**Files:**
- Modify (generated): `out/sft/` (adapter), backup `out/sft_11ex_bak/`

**Interfaces:**
- Consumes: `data/sft_v3/train.jsonl`; existing `autotune/train_sft.py` (CUDA backend, `train.jsonl` with `messages` rows).
- Produces: retrained adapter at `out/sft/` for Task 10/11.

- [ ] **Step 1: Verify GPU** — run `nvidia-smi`; expect the RTX 5090 listed. If not, stop and tell the user to run /verify-nvidia.

- [ ] **Step 2: Back up the current adapter**

```bash
mv out/sft out/sft_11ex_bak
```

- [ ] **Step 3: Train**

```bash
uv run python -m autotune.train_sft --data-dir data/sft_v3 2>&1 | tee out/sft_v3_train.log
```

Expected: `[train_sft] loaded ~N SFT examples` where N matches stats.json train count; loss decreasing but NOT collapsing to ~0.003 like the 11-example run (healthy end-of-training loss on a few-thousand-example corpus is roughly 0.5–1.5). Runtime: expect tens of minutes.

- [ ] **Step 4: Commit the log reference (not weights)**

```bash
git add -f out/sft_v3_train.log && git commit -m "train: LoRA SFT on sft_v3 corpus (5090)"
```

---

### Task 10: held-out eval gate (tuned vs base)

**Files:**
- Create: `autotune/eval_heldout.py`
- Create: `tests/test_eval_heldout.py`

**Interfaces:**
- Consumes: `data/sft_v3/valid.jsonl` (rows keep `domain`), adapter at `out/sft/`, base `HuggingFaceTB/SmolLM3-3B`.
- Produces: `parse_json_answer(text: str) -> dict | None`; `score_rows(rows: list[dict], predict) -> dict[str, float]` (accuracy per gated domain, keys `battle-outcome`, `move-choice`); CLI writing `out/eval/heldout.json` `{"base": {...}, "tuned": {...}, "passed": bool}`, exit 0 iff tuned beats base on BOTH gated domains.

- [ ] **Step 1: Failing tests** (pure functions only — no GPU in unit tests)

`tests/test_eval_heldout.py`:

```python
import json

from autotune.eval_heldout import parse_json_answer, score_rows


def test_parse_json_answer_extracts_first_object():
    assert parse_json_answer('noise {"win": true} trailing') == {"win": True}
    assert parse_json_answer("no json here") is None


def _row(domain, user, answer):
    return {
        "domain": domain,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def test_score_rows_accuracy_per_domain():
    rows = [
        _row("battle-outcome", "u1", json.dumps({"win": True, "recommendation": "fight"})),
        _row("battle-outcome", "u2", json.dumps({"win": False, "recommendation": "flee"})),
        _row("move-choice", "u3", json.dumps({"move": "Ember"})),
        _row("narrator", "u4", "not gated"),
    ]
    # fake model always answers win=True / move=Ember
    def predict(system, user):
        return '{"win": true, "recommendation": "fight", "move": "Ember"}'
    scores = score_rows(rows, predict)
    assert scores == {"battle-outcome": 0.5, "move-choice": 1.0}
```

- [ ] **Step 2: Verify failure.** `uv run python -m pytest tests/test_eval_heldout.py -v` → ImportError.

- [ ] **Step 3: Implement**

`autotune/eval_heldout.py`:

```python
"""Held-out eval gate: tuned model must beat base SmolLM3-3B on ground-truth domains.

Scores battle-outcome (win-field accuracy) and move-choice (move/bucket-field accuracy) on
data/sft_v3/valid.jsonl. ``parse_json_answer``/``score_rows`` are pure and unit-tested; the
generation loop is a GPU wrapper exercised manually (like train_sft/package).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GATED_DOMAINS = ("battle-outcome", "move-choice")
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_answer(text: str) -> dict | None:
    """First {...} object in text, or None."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _match(expected: dict, got: dict | None) -> bool:
    if got is None:
        return False
    for key in ("win", "move", "bucket"):
        if key in expected:
            return expected.get(key) == got.get(key)
    return False


def score_rows(rows: list[dict], predict) -> dict[str, float]:
    """Accuracy per gated domain. ``predict(system, user) -> str``."""
    hits: dict[str, list[bool]] = {}
    for row in rows:
        domain = row.get("domain")
        if domain not in GATED_DOMAINS:
            continue
        system, user = row["messages"][0]["content"], row["messages"][1]["content"]
        expected = json.loads(row["messages"][2]["content"])
        got = parse_json_answer(predict(system, user))
        hits.setdefault(domain, []).append(_match(expected, got))
    return {d: sum(v) / len(v) for d, v in hits.items()}


def _hf_predictor(model_id: str, adapter: str | None):  # pragma: no cover - GPU wrapper
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def predict(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return predict


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Held-out eval: tuned vs base.")
    p.add_argument("--valid", type=Path, default=Path("data/sft_v3/valid.jsonl"))
    p.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B")
    p.add_argument("--adapter", default="out/sft")
    p.add_argument("--out", type=Path, default=Path("out/eval/heldout.json"))
    p.add_argument("--limit", type=int, default=None, help="cap rows per domain for a quick pass")
    args = p.parse_args(argv)

    rows = [json.loads(x) for x in args.valid.read_text().splitlines() if x.strip()]
    if args.limit:
        capped: dict[str, int] = {}
        rows = [r for r in rows if capped.setdefault(r.get("domain"), 0) < args.limit
                and not capped.update({r.get("domain"): capped[r.get("domain")] + 1})]

    base_scores = score_rows(rows, _hf_predictor(args.base, None))
    tuned_scores = score_rows(rows, _hf_predictor(args.base, args.adapter))
    passed = all(tuned_scores.get(d, 0) > base_scores.get(d, 0) for d in GATED_DOMAINS)
    result = {"base": base_scores, "tuned": tuned_scores, "passed": passed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Unit tests pass.** `uv run python -m pytest tests/test_eval_heldout.py -v`

- [ ] **Step 5: Run the real eval on the GPU**

```bash
uv run python -m autotune.eval_heldout 2>&1 | tee out/eval/heldout.log
```

Expected: exit 0 with `"passed": true` and tuned > base on both domains. **If it exits 1: STOP — do not proceed to Task 11/12.** Report the numbers to the user; likely follow-ups are more training steps or corpus fixes. Publishing is gated on this.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && git add autotune/eval_heldout.py tests/test_eval_heldout.py out/eval/heldout.json && git commit -m "feat(eval): held-out tuned-vs-base gate on ground-truth domains"
```

---

### Task 11: fuse + manifest

**Files:**
- Modify (generated): `out/package/` (backup old one first)

**Interfaces:**
- Consumes: adapter at `out/sft/`, existing `autotune/package.py` (`--skip-eval` skips its game-rollout eval; our gate was Task 10).
- Produces: `out/package/fused/` + `out/package/manifest.json` for Task 12.

- [ ] **Step 1: Back up the old package**

```bash
mv out/package out/package_11ex_bak
```

- [ ] **Step 2: Fuse**

```bash
uv run python -m autotune.package --skip-eval --seed 42 2>&1 | tee out/package_fuse.log
```

Expected: `out/package/fused/` with `model.safetensors`, config, tokenizer; `manifest.json` with `base_model: HuggingFaceTB/SmolLM3-3B` and a fresh `train_data_sha256_16` matching `data/sft_v3`.

- [ ] **Step 3: Smoke the fused model**

```bash
uv run python - <<'EOF'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("out/package/fused")
model = AutoModelForCausalLM.from_pretrained("out/package/fused", torch_dtype=torch.bfloat16, device_map="auto")
msgs = [{"role": "system", "content": "You are the battle advisor for a Pokemon Red agent. Answer with only the requested JSON."},
        {"role": "user", "content": 'Battle start.\nYour Pokemon: Charmander (lv 6, HP 21/21), move types: normal, fire.\nEnemy: Weedle (lv 3, bug type). Level gap: +3. Healing available: no.\nWill the agent win this battle, and should it fight or flee? Respond with JSON {"win": bool, "recommendation": "fight"|"flee"}.'}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
EOF
```

Expected: a JSON answer in the trained shape (e.g. `{"win": true, "recommendation": "fight"}`).

- [ ] **Step 4: Commit manifest**

```bash
git add -f out/package/manifest.json out/package_fuse.log && git commit -m "package: fuse sft_v3 adapter into SmolLM3-3B"
```

---

### Task 12: model card + publish to Hugging Face Hub

**Files:**
- Create: `out/package/fused/README.md` (model card)
- Create: `data/sft_v3/README.md` (dataset card)

**Interfaces:**
- Consumes: `out/package/fused/`, `out/sft/` adapter, `data/sft_v3/`, `out/eval/heldout.json` numbers.
- Produces: public HF repos — model (fused at root, adapter under `adapter/`) and dataset.

- [ ] **Step 1: Write the model card**

`out/package/fused/README.md` (fill the eval table from `out/eval/heldout.json` before uploading):

```markdown
---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM3-3B
tags: [pokemon, game-agent, lora, smollm3]
---

# SmolLM3-3B Pokemon Red Agent (multi-task)

SmolLM3-3B LoRA-fine-tuned (fused) on telemetry from an autonomous Pokemon Red agent
([pokemon-kafka](https://github.com/pcc-labs/pokemon-kafka) + empirical-evidence training loop).

Three tasks in one model, all prompted with plain chat:
- **Battle advisor** — pre-battle state → `{"win": bool, "recommendation": "fight"|"flee"}`; move
  damage buckets and best-move picks (ground-truth labels from game RAM telemetry).
- **Genome proposer** — scenario + fitness summary → agent survival-parameter JSON.
- **Narrator** — game event JSON → one-sentence play-by-play.

## Held-out eval (vs base SmolLM3-3B)

| Domain | Base | Tuned |
|---|---|---|
| battle-outcome (win prediction) | FILL | FILL |
| move-choice (type effectiveness) | FILL | FILL |

## Training data

~N examples (see dataset repo) generated deterministically (seed 42) from ~72k
`pokemon.game.v1` telemetry events. Prompts are template-generated from game state — this is a
demo-scale agent assistant, not a general Pokemon expert. Corpus SHA and per-domain counts are in
`manifest.json` / the dataset's `stats.json`.

## Limitations

- Trained only on early-game Pokemon Red states (Pallet → Pewter); species/type coverage is narrow.
- Narrator labels are synthetic (template pools), so narration is stylistic, not learned commentary.
- The genome task is agent-specific; genome JSON is meaningless outside the pokemon-kafka agent.
```

Also write `data/sft_v3/README.md` with the dataset YAML header (`license: apache-2.0`, task_categories `text-generation`), the five-domain table from `stats.json`, and the converter command line.

- [ ] **Step 2: Verify HF auth**

Run: `uv run hf auth whoami`
Expected: a logged-in username. If not authenticated, ask the user to run `! uv run hf auth login`.

- [ ] **Step 3: USER GATE — confirm repo names**

Propose to the user: model repo `<username>/smollm3-3b-pokemon-red-agent`, dataset repo `<username>/pokemon-red-telemetry-sft`. **Do not upload until the user confirms names and public visibility.**

- [ ] **Step 4: Upload**

```bash
uv run hf upload <user>/smollm3-3b-pokemon-red-agent out/package/fused . --repo-type model --commit-message "fused SmolLM3-3B pokemon multi-task v1"
uv run hf upload <user>/smollm3-3b-pokemon-red-agent out/sft adapter --repo-type model --commit-message "LoRA adapter v1"
uv run hf upload <user>/pokemon-red-telemetry-sft data/sft_v3 . --repo-type dataset --commit-message "sft_v3 corpus (seed 42)"
```

- [ ] **Step 5: Verify hosted model**

```bash
uv run python - <<'EOF'
from huggingface_hub import model_info
info = model_info("<user>/smollm3-3b-pokemon-red-agent")
print(info.modelId, [s.rfilename for s in info.siblings][:10])
EOF
```

Expected: repo exists and lists `model.safetensors`, `README.md`, `adapter/adapter_model.safetensors`.

- [ ] **Step 6: Commit cards + final state**

```bash
git add data/sft_v3/README.md && git commit -m "docs: model + dataset cards for HF release"
```

---

## Self-review notes

- Spec coverage: loader/error handling (T1, T7), five generators (T2–T6), dedup/balance/split/stats/hard-fail (T7), real-corpus run (T8), 5090 SFT (T9), eval gate incl. "don't publish if not better than base" (T10 Step 5 STOP), fuse+manifest (T11), model card + Hub publish (T12). DPO/Kafka/LLM-paraphrase correctly absent (out of scope).
- Deviation from spec recorded in Global Constraints: module path is `autotune/convert_telemetry.py`, not `scripts/`.
- Type consistency: `chat()` example dict shape is identical across all generators and the eval module's `_row` fixture; `rng: random.Random` threaded through `gen_battle_action`, `gen_narrator`, `balance`, `split`.
