"""Convert pokemon.game.v1 telemetry into a multi-domain SFT corpus.

Deterministic (seeded) converter per
docs/superpowers/specs/2026-07-05-telemetry-sft-converter-design.md.
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


BATTLE_SYSTEM = (
    "You are the battle advisor for a Pokemon Red agent. Answer with only the requested JSON."
)


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
            f"Enemy: {d['enemy_species']} (lv {d['enemy_level']}, "
            f"{d['enemy_type']} type). "
            f"Level gap: {d['level_gap']:+d}. "
            f"Healing available: {'yes' if d['had_healing'] else 'no'}.\n"
            "Will the agent win this battle, and should it fight or flee? "
            'Respond with JSON {"win": bool, "recommendation": "fight"|"flee"}.'
        )
        answer = json.dumps({"win": d["won"], "recommendation": "fight" if d["won"] else "flee"})
        out.append(chat(BATTLE_SYSTEM, user, answer, "battle-outcome"))
    return out
