"""npc-dialogue: the Forger's domain.

One example per body the crew talked to (``supervisor.body_engaged`` and
``supervisor.blocker_engaged`` in pokemon-kafka's expedition sink). The prompt is where the body
stands and what it said, read from the screen; the label is what the body is (the cartridge's
sprite table) and what came of the talk, measured from the same run: an item in the bag
(``gained``), a fight the talk started (the
``battle.outcome`` / ``battle.fled`` row just before the engage row), a gate sentence (the measured
gate classes), a stale window (the same sentence at three or more cells in one run), or plain talk.
Deterministic; nothing recalled.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from autotune.convert_telemetry import chat
from autotune.handoff_corpus import classify_gate

NPC_DOMAIN = "npc-dialogue"
NPC_SYSTEM = (
    "You are the Forger for a Pokemon Red crew: you read what a body says when the crew talks "
    "to it and decide what that body is worth. Answer with only the requested JSON."
)
ENGAGE = "supervisor.body_engaged"
BLOCKER = "supervisor.blocker_engaged"
LEG_START = "supervisor.leg_start"
FIGHTS = ("battle.outcome", "battle.fled")
FIGHT_WINDOW_S = 5.0  # the talk starts the fight; its outcome row lands just before the engage row
STALE_MIN_CELLS = 3  # one sentence read at this many cells of one run is the window, not the body
# Reads that are not the body's words: the START menu the engage loop opens on an item-ball tile
# (measured on maps 194, 219, 234) and the battle text pinned after a flee.
NOISE = frozenset({"OPTION EXIT", "Got away safely!"})
BODIES = ("trainer", "npc", "item", "unknown")
OUTCOMES = ("talk", "handed", "fought-won", "fought-lost", "fled", "gate", "blocker", "stale")


def clean_said(said: str) -> str:
    """One sentence from the decoder's growing window reads: partial reads dropped, the overlap
    between consecutive reads merged once."""
    parts = [p.strip() for p in said.split("|") if p.strip()]
    keep: list[str] = []
    for p in parts:
        if any(o != p and o.startswith(p) for o in parts):
            continue
        if keep and keep[-1] == p:
            continue
        keep.append(p)
    out = ""
    for p in keep:
        if not out:
            out = p
            continue
        k = next((n for n in range(min(len(out), len(p)), 2, -1) if out.endswith(p[:n])), 0)
        out = out + p[k:] if k else out + " " + p
    return out


def _ts(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def load_truth(pk_root: Path) -> tuple[dict, dict]:
    """(sprite index keyed (map, x, y), item names by id) from pokemon-kafka's extracted ROM
    truth; empty when the checkout has none (rows then carry body "unknown")."""
    path = pk_root / "references" / "rom_truth.json"
    if not path.exists():
        print(f"[npc] warning: no rom truth at {path}; bodies will be unknown")
        return {}, {}
    truth = json.loads(path.read_text())
    sprites = {}
    for mid, m in truth.get("maps", {}).items():
        for s in m.get("sprites") or []:
            sprites[(int(mid), int(s["x"]), int(s["y"]))] = s
    return sprites, truth.get("items", {})


def _stale_sentences(events: list[dict]) -> set[tuple[str, str]]:
    """(run_id, sentence) pairs read at STALE_MIN_CELLS or more distinct cells of one run."""
    cells: dict[tuple[str, str], set[tuple]] = {}
    for e in events:
        if e.get("event") != ENGAGE:
            continue
        sentence = clean_said(e.get("said") or "")
        at = tuple(e.get("at") or ())
        if sentence and at:
            cells.setdefault((e.get("run_id", ""), sentence), set()).add(at)
    return {k for k, v in cells.items() if len(v) >= STALE_MIN_CELLS}


def gen_npc_dialogue(events: list[dict], sprites: dict, items: dict) -> list[dict]:
    out: list[dict] = []
    stale = _stale_sentences(events)
    cur_map: dict[str, int] = {}
    last_fight: dict[str, tuple[datetime | None, bool | None]] = {}
    for e in events:
        run = e.get("run_id", "")
        kind = e.get("event")
        if kind == LEG_START:
            pos = e.get("pos") or []
            if pos:
                cur_map[run] = int(pos[0])
            continue
        if kind in FIGHTS:
            last_fight[run] = (
                _ts(e.get("ts")),
                e.get("won") if kind == "battle.outcome" else None,
            )
            continue
        if kind not in (ENGAGE, BLOCKER):
            continue
        if kind == ENGAGE:
            mp = e.get("map")
            x, y = (e.get("at") or [None, None])[:2]
        else:
            mp = cur_map.get(run)
            x, y = (e.get("body") or [None, None])[:2]
        if mp is None or x is None:
            continue
        cur_map[run] = int(mp)
        sentence = clean_said(e.get("said") or "")
        gained = [items.get(str(i), f"#{i}") for i, _q in (e.get("gained") or [])]
        if sentence in NOISE:
            sentence = ""
        if not sentence and not gained:
            continue
        sprite = sprites.get((int(mp), int(x), int(y))) or {}
        body = sprite.get("kind", "unknown")
        when, won = last_fight.get(run, (None, None))
        now = _ts(e.get("ts"))
        fought = bool(when and now and 0 <= (now - when).total_seconds() <= FIGHT_WINDOW_S)
        if fought:
            last_fight.pop(run, None)  # one fight, one body
        hit = classify_gate(sentence)
        if gained:
            outcome = "handed"
        elif fought:
            # battle.outcome carries won; battle.fled carries no verdict (won is None)
            outcome = "fought-won" if won else ("fought-lost" if won is not None else "fled")
        elif hit:
            outcome = "gate"
        elif kind == BLOCKER:
            outcome = "blocker"
        elif (run, sentence) in stale:
            outcome = "stale"
        else:
            outcome = "talk"
        pic = f" (sprite pic {sprite['pic']})" if sprite.get("pic") is not None else ""
        user = (
            f"On map {mp}, the crew talked to the body at ({x}, {y}){pic}. "
            f"It said: {sentence!r}.\n"
            "What is this body, and what comes of talking to it? Respond with JSON "
            '{"body": "trainer"|"npc"|"item"|"unknown", "outcome": "talk"|"handed"|'
            '"fought-won"|"fought-lost"|"fled"|"gate"|"blocker"|"stale", '
            '"items": [str], "gate": str|null}'
        )
        label = {"body": body, "outcome": outcome, "items": gained, "gate": hit[0] if hit else None}
        row = chat(NPC_SYSTEM, user, json.dumps(label, sort_keys=True), NPC_DOMAIN, e)
        row["meta"] = {
            **row.get("meta", {}),
            "map": int(mp),
            "x": x,
            "y": y,
            "event": kind,
            "pic": sprite.get("pic"),
        }
        out.append(row)
    return out
