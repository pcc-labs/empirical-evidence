"""Handoff domains: where the crew stopped and a human had to step in.

Three generators over pokemon-kafka's expedition event sink (``data/telemetry/game/*.jsonl``,
events with ``"source": "expedition"``) and its written exhaustion records
(``docs/learnings/map<A>-to-<B>-stuck-<run>.md``):

* ``handoff`` -- one example per ladder exhaustion or item/badge hunt that ended without the
  prize. The prompt is the supervisor's own measured-facts block; the answer is what the human
  operator measured and did next, curated in ``handoff_resolutions.json`` with the learnings doc
  or commit that records it. These labels are human-written, not RAM-derived.
* ``puzzle-consult`` -- one example per model consult at a wall (the Point Man / Extractor seats).
  The prompt carries the wall, the failure code, the menu and the seat's chosen action + reason;
  the label is what the engine returned when that action was executed, read from the next event
  of the same run. Ground truth from the engine, not annotation.
* ``gate-text`` -- one example per refusal sentence the game put on screen at a measured cell
  ("This requires STRENGTH", "No SURFing on GYARADOS here!", "The current is much too fast!",
  ...). The label is the gate class and the verb that was measured to clear it.

Everything is deterministic: files are read in sorted order and no randomness is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from autotune.convert_telemetry import before_since, chat
from autotune.memlog import MemLog

HANDOFF_DOMAIN = "handoff"
CONSULT_DOMAIN = "puzzle-consult"
GATE_DOMAIN = "gate-text"
DOMAINS = (HANDOFF_DOMAIN, CONSULT_DOMAIN, GATE_DOMAIN)

WANTED_EVENTS = frozenset(
    {
        "supervisor.leg_start",
        "supervisor.hop_failed",
        "supervisor.consult",
        "supervisor.exhausted",
        "supervisor.leg_end",
        "supervisor.rerouted",
        "supervisor.gate_text",
        "supervisor.body_engaged",
        "supervisor.blocker_engaged",
        "milestone",
        "discovery",
        "refusal",
    }
)

# The supervisor's bounded menus (scripts/supervisor.py MENUS / DEFAULT_MENU). The event sink
# records the seat's choice but not the menu it was offered, so the menu is rebuilt from the
# failure code; a chosen action outside it (ORACLE_SEARCH / SWEEP_ITEMS are inserted only on
# facility floors) is appended so the prompt never shows an answer off its own menu.
MENUS: dict[str, tuple[str, ...]] = {
    "no-route": ("BACK_OUT_AND_REENTER", "TALK_TO_BLOCKER", "USE_GATE_WARP", "GIVE_UP"),
    "no-path": (
        "USE_GATE_WARP",
        "TRY_FAR_EDGE_CELL",
        "WAIT_FOR_BODIES",
        "BACK_OUT_AND_REENTER",
        "GIVE_UP",
    ),
    "body-blocked": (
        "WAIT_FOR_BODIES",
        "TALK_TO_BLOCKER",
        "TRY_FAR_EDGE_CELL",
        "RETRY_SAME",
        "GIVE_UP",
    ),
    "refused": (
        "TALK_TO_BLOCKER",
        "BACK_OUT_AND_REENTER",
        "TRY_FAR_EDGE_CELL",
        "RETRY_SAME",
        "GIVE_UP",
    ),
    "stuck-on-edge": (
        "TRY_FAR_EDGE_CELL",
        "USE_GATE_WARP",
        "RETRY_SAME",
        "BACK_OUT_AND_REENTER",
        "GIVE_UP",
    ),
}
DEFAULT_MENU = (
    "RETRY_SAME",
    "TRY_FAR_EDGE_CELL",
    "USE_GATE_WARP",
    "BACK_OUT_AND_REENTER",
    "GIVE_UP",
)

# Gate sentences the game put on screen, and what was measured to clear each. The regex is the
# sentence; the class and the verb come from docs/learnings (cited in the dataset card), never
# from recalled game lore.
GATES: tuple[tuple[str, str, str], ...] = (
    (
        "strength_boulder",
        r"requires STRENGTH",
        "STRENGTH: the sprite is a boulder (pic 63), not a body; push it with the field move "
        "(a 16-frame hold) and read the sprite table for the verdict",
    ),
    (
        "surf_launch_refused",
        r"No SURFing on",
        "SURF, armed through the POKeMON menu (use_field_move by species) from a shore cell that "
        "touches edge-reaching water, facing the water; the refusal is the facing, not the menu",
    ),
    (
        "surf_no_landing",
        r"no place to get off",
        "you are already on the water facing a cell with no landing; turn toward land or keep "
        "surfing along the water component",
    ),
    (
        "current_too_fast",
        r"current is much too fast",
        "a boulder dropped through a 0x22 hole on the floor above must land in this channel "
        "before the surf is accepted; solve the boulder chain, not the launch cell",
    ),
    (
        "card_key_door",
        r"Darn! It needs a CARD KEY",
        "the CARD KEY from Silph 5F's item ball at (21,16), reached by riding the (27,3) pad and "
        "stepping back; the door opens on the press once it is in the bag",
    ),
    (
        "sleeping_blocker",
        r"sleeping POK",
        "the POKe FLUTE from the bag, played while facing the sleeper; the vacated sprite slot "
        "still reads as a body afterwards but no longer blocks",
    ),
    (
        "route_gate_guard",
        r"Wait up please",
        "nothing from inside: the gate's lower corridor is sealed from its upper one; go back "
        "out, CUT the bush, and enter the upper corridor through the upper doors",
    ),
    (
        "script_guard",
        r"Get out of the way",
        "a story gate, not a trainer: beat Giovanni on Silph 11F and the guard stands down "
        '("We admire your courage.")',
    ),
    (
        "trainer_challenge",
        r"heart attack",
        "a trainer: the battle opens on the talk; win it and the body is no longer in the way",
    ),
    (
        "nothing_to_cut",
        r"isn't anything to CUT",
        "CUT only clears a 0x3D bush; stand adjacent to the bush and face it before using the move",
    ),
    (
        "stale_window_text",
        r"hope to see you again",
        "not a gate: the shop's farewell left on the sticky window layer; a text box blocks "
        "movement, so gate the read on probe_step and consult the collision grid for the wall",
    ),
)
_GATE_PATTERNS = tuple((cls, re.compile(pat), clears) for cls, pat, clears in GATES)
_GATE_PREFILTER = re.compile("|".join(pat for _, pat, _ in GATES))
_EVENT_RE = re.compile(r'"event": "([^"]+)"')


def load_expedition_events(
    game_dir: Path, memlog: MemLog | None = None, since: str | None = None
) -> tuple[list[dict], int]:
    """Read the expedition events the generators use, in file order. Returns (events, skipped).

    The sink also carries a million-line ``discovery``/``kind: discovery`` firehose and ~30k
    ordinary NPC lines; both are dropped on the raw line before any JSON is parsed. With
    ``memlog``, one memory record per file (lines seen, events kept, rss after).
    """
    events: list[dict] = []
    skipped = 0
    if not game_dir.exists():
        print(f"[handoff] warning: missing expedition sink {game_dir}")
        return events, skipped
    for path in sorted(game_dir.glob("*.jsonl")):
        lines = kept = 0
        with open(path) as f:
            for line in f:
                lines += 1
                if '"source": "expedition"' not in line or '"kind": "discovery"' in line:
                    continue
                if before_since(line, since):
                    continue
                m = _EVENT_RE.search(line)
                if not m or m.group(1) not in WANTED_EVENTS:
                    continue
                if m.group(1) in (
                    "supervisor.body_engaged",
                    "refusal",
                ) and not _GATE_PREFILTER.search(line):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(d, dict):
                    events.append(d)
                    kept += 1
        if memlog:
            memlog.note_file(str(path), kept)
            memlog.log(
                "load_expedition_events.file",
                file=str(path),
                bytes=path.stat().st_size,
                lines=lines,
                kept=kept,
                events=len(events),
            )
    return events, skipped


# ---------------------------------------------------------------------------
# Stuck docs and the resolution table
# ---------------------------------------------------------------------------


def parse_stuck_doc(path: Path) -> dict:
    """The supervisor's exhaustion record: the fenced facts block and the actions tried."""
    text = path.read_text()
    facts = ""
    m = re.search(r"## Measured facts at the point of failure\s*```\n(.*?)```", text, re.S)
    if m:
        facts = m.group(1).strip()
    tried: list[str] = []
    m = re.search(r"## Actions tried\s*\n(.*?)(?:\n## |\Z)", text, re.S)
    if m:
        tried = [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")]
    return {"facts": facts, "tried": tried}


def load_resolutions(path: Path) -> list[dict]:
    return json.loads(path.read_text())["resolutions"]


def _in(rule_value, actual) -> bool:
    if rule_value is None:
        return True
    if isinstance(rule_value, list):
        return actual in rule_value
    return actual == rule_value


def _matches(m: dict, *, map_id, goal, failure, outcome, reason) -> bool:
    """Every stated key of one match clause holds."""
    if not _in(m.get("map"), map_id):
        return False
    if not _in(m.get("goal"), goal):
        return False
    if not _in(m.get("failure"), failure):
        return False
    if not _in(m.get("outcome"), outcome):
        return False
    if m.get("reason_contains") and m["reason_contains"] not in (reason or ""):
        return False
    return True


def match_resolution(rules: list[dict], *, map_id, goal, failure, outcome, reason) -> dict | None:
    """First rule with a matching clause; rules are ordered most specific first.

    A rule's ``match`` is one clause (dict) or a list of alternative clauses (any may match).
    """
    for rule in rules:
        clauses = rule["match"] if isinstance(rule["match"], list) else [rule["match"]]
        for m in clauses:
            if _matches(
                m, map_id=map_id, goal=goal, failure=failure, outcome=outcome, reason=reason
            ):
                return rule
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HANDOFF_SYSTEM = (
    "You are the human operator of a Pokemon Red expedition. The crew (local models seated by a "
    "router) exhausted its bounded action ladder and handed you its written record. Nothing "
    "recalled about the game may be load-bearing: answer from the measured facts and from what "
    "this cartridge has already been measured to do. "
    'Reply with JSON {"diagnosis": str, "do": [str, ...]}.'
)
CONSULT_SYSTEM = (
    "You are reviewing a Pokemon Red expedition crew's decision at a wall. The road engine will "
    "execute the chosen action; predict what it returns. Answer with only the requested JSON."
)
GATE_SYSTEM = (
    "You are the navigation advisor for a Pokemon Red expedition agent. What the game says on "
    "screen is the instruction stream. Answer with only the requested JSON."
)


def _by_run(events: list[dict]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for e in events:
        if e.get("event", "").startswith("supervisor.") or e.get("event") == "milestone":
            runs.setdefault(e.get("run_id", ""), []).append(e)
    return runs


def _doc_for(exhausted: dict, learnings_dir: Path) -> Path | None:
    """The stuck doc, resolved by basename under this checkout.

    The event stores an absolute path.
    """
    doc = exhausted.get("doc")
    if not doc:
        return None
    path = learnings_dir / Path(doc).name
    return path if path.exists() else None


def _facts_from_event(e: dict, last_wall: str | None) -> str:
    mp, x, y = e["pos"]
    lines = [f"GOAL: reach map {e['goal']}. You are on map {mp} at ({x}, {y})."]
    if last_wall:
        lines.append(f"FAILED HOP: {last_wall}; the engine returned {e.get('failure')!r}.")
    else:
        lines.append(f"The engine returned {e.get('failure')!r}.")
    return "\n".join(lines)


def _handoff_user(facts: str, tried: list[str], ending: str) -> str:
    tried_text = "\n".join(f"- {t}" for t in tried) if tried else "- (none)"
    return (
        f"{ending}\n\n## Measured facts at the point of failure\n\n{facts}\n\n"
        f"## Actions tried\n\n{tried_text}\n\nWhat do you measure or do next?"
    )


def gen_handoff(
    events: list[dict], learnings_dir: Path, rules: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Handoff rows; also returns the unmatched handoffs so a run can report what has no answer."""
    out: list[dict] = []
    unmatched: list[dict] = []
    runs = _by_run(events)
    for run_id in sorted(runs):
        evs = runs[run_id]
        last_wall = None
        exhausted_goals: set = set()
        for e in evs:
            kind = e.get("event")
            if kind == "supervisor.hop_failed":
                last_wall = e.get("wall")
            elif kind == "supervisor.exhausted":
                mp = e["pos"][0]
                doc = _doc_for(e, learnings_dir)
                parsed = (
                    parse_stuck_doc(doc)
                    if doc
                    else {"facts": "", "tried": list(e.get("tried") or [])}
                )
                facts = parsed["facts"] or _facts_from_event(e, last_wall)
                tried = parsed["tried"] or list(e.get("tried") or [])
                rule = match_resolution(
                    rules,
                    map_id=mp,
                    goal=e["goal"],
                    failure=e.get("failure"),
                    outcome="exhausted",
                    reason="",
                )
                exhausted_goals.add(e["goal"])
                meta = {
                    "run_id": run_id,
                    "map": mp,
                    "goal": e["goal"],
                    "failure": e.get("failure"),
                    "doc": doc.name if doc else None,
                }
                if not rule:
                    unmatched.append(meta)
                    continue
                ending = (
                    f"The ladder was exhausted on map {mp} toward map {e['goal']}: "
                    "the Point Man then the "
                    f"Extractor, and the engine's last word was {e.get('failure')!r}."
                )
                row = chat(
                    HANDOFF_SYSTEM,
                    _handoff_user(facts, tried, ending),
                    json.dumps(rule["resolution"]),
                    HANDOFF_DOMAIN,
                )
                row["meta"] = {**meta, "sources": rule["sources"]}
                out.append(row)
            elif kind == "supervisor.leg_end" and not e.get("ok"):
                outcome = e.get("outcome")
                if (
                    outcome not in ("engaged-no-item", "engaged-no-badge")
                    or e["goal"] in exhausted_goals
                ):
                    continue
                mp, x, y = e["pos"]
                rule = match_resolution(
                    rules,
                    map_id=mp,
                    goal=e["goal"],
                    failure=None,
                    outcome=outcome,
                    reason=e.get("reason"),
                )
                meta = {
                    "run_id": run_id,
                    "map": mp,
                    "goal": e["goal"],
                    "failure": outcome,
                    "doc": None,
                }
                if not rule:
                    unmatched.append(meta)
                    continue
                facts = (
                    f"GOAL: reach map {e['goal']} and come away with the prize. "
                    f"You are on map {mp} at ({x}, {y}).\n"
                    f"LEG OUTCOME: {outcome} -- {e.get('reason')}.\nBADGES byte: {e.get('badges')}"
                )
                ending = f"The leg ended without the prize: {e.get('reason')}."
                row = chat(
                    HANDOFF_SYSTEM,
                    _handoff_user(facts, [], ending),
                    json.dumps(rule["resolution"]),
                    HANDOFF_DOMAIN,
                )
                row["meta"] = {**meta, "sources": rule["sources"]}
                out.append(row)
    return out, unmatched


def _consult_outcome(evs: list[dict], i: int, wall: str) -> tuple[bool, str] | None:
    """What the engine did after the consult at evs[i]: measured from the next informative event."""
    for f in evs[i + 1 :]:
        kind = f.get("event")
        if kind == "supervisor.hop_failed":
            if f.get("wall") == wall:
                return False, f.get("failure", "failed")
            return True, "ok"
        if kind == "supervisor.leg_end":
            return (True, "arrived") if f.get("ok") else (False, f.get("outcome", "failed"))
        if kind == "supervisor.exhausted":
            return False, "exhausted"
        if kind == "supervisor.rerouted":
            return False, "rerouted"
        if kind == "milestone":
            return True, "milestone"
        if kind == "supervisor.consult":
            return False, "retried"
    return None


def gen_puzzle_consult(events: list[dict], learnings_dir: Path) -> list[dict]:
    out: list[dict] = []
    runs = _by_run(events)
    for run_id in sorted(runs):
        evs = runs[run_id]
        goal = None
        facts_by_wall: dict[str, str] = {}
        for e in evs:
            if e.get("event") == "supervisor.exhausted":
                doc = _doc_for(e, learnings_dir)
                if doc:
                    facts = parse_stuck_doc(doc)["facts"]
                    m = re.search(r"FAILED HOP: (\d+) --\w+--> (\d+)", facts)
                    if m:
                        facts_by_wall[f"{m.group(1)}->{m.group(2)}"] = facts
        attempts: dict[str, int] = {}
        tried: list[str] = []
        for i, e in enumerate(evs):
            kind = e.get("event")
            if kind == "supervisor.leg_start":
                goal = e.get("goal")
            elif kind == "supervisor.hop_failed":
                attempts[e.get("wall", "")] = e.get(
                    "attempt", attempts.get(e.get("wall", ""), 0) + 1
                )
            elif kind == "supervisor.consult":
                if e.get("model") in (None, "", "none") or not e.get("action"):
                    continue
                wall = e.get("wall", "")
                failure = None
                for f in reversed(evs[:i]):
                    if f.get("event") == "supervisor.hop_failed" and f.get("wall") == wall:
                        failure = f.get("failure")
                        break
                outcome = _consult_outcome(evs, i, wall)
                if outcome is None:
                    continue
                menu = list(MENUS.get(failure or "", DEFAULT_MENU))
                if e["action"] not in menu:
                    menu.insert(0, e["action"])
                facts = facts_by_wall.get(wall)
                if not facts:
                    src = wall.split("->")[0]
                    facts = f"GOAL: reach map {goal}. You are on map {src}, at the {wall} hop."
                user = (
                    f"{facts}\n"
                    f"FAILED HOP: {wall}; the engine returned {failure!r} "
                    f"(attempt {attempts.get(wall, 1)}).\n"
                    f"ACTIONS ALREADY TRIED THIS LEG: {tried if tried else '[]'}\n"
                    f"MENU: {', '.join(menu)}\n"
                    f"The {e.get('tier')} seat ({e.get('model')}) chose:\n"
                    f"ACTION: {e['action']}\nWHY: {e.get('why', '').strip()}\n"
                    "Does this action advance the leg? "
                    'Respond with JSON {"advances": bool, "engine_returned": str}.'
                )
                advances, returned = outcome
                row = chat(
                    CONSULT_SYSTEM,
                    user,
                    json.dumps({"advances": advances, "engine_returned": returned}),
                    CONSULT_DOMAIN,
                )
                row["meta"] = {
                    "run_id": run_id,
                    "wall": wall,
                    "tier": e.get("tier"),
                    "model": e.get("model"),
                    "action": e["action"],
                    "failure": failure,
                }
                out.append(row)
                tried.append(f"{e['action']} on {wall}")
    return out


def classify_gate(text: str) -> tuple[str, str] | None:
    for cls, pat, clears in _GATE_PATTERNS:
        if pat.search(text):
            return cls, clears
    return None


def gen_gate_text(events: list[dict]) -> list[dict]:
    out: list[dict] = []
    for e in events:
        kind = e.get("event")
        if kind not in (
            "discovery",
            "supervisor.gate_text",
            "supervisor.body_engaged",
            "supervisor.blocker_engaged",
            "refusal",
        ):
            continue
        text = e.get("said") or e.get("text") or ""
        hit = classify_gate(text)
        if not hit:
            continue
        cls, clears = hit
        # The screen decoder repeats a growing sentence; keep its longest clause.
        sentence = max((s.strip() for s in text.split("|")), key=len)
        if kind == "refusal":
            mp, x, y = e.get("from"), None, None
        else:
            mp = e.get("map")
            x, y = (e.get("at") or [e.get("x"), e.get("y")])[:2]
        where = f"On map {mp}" + (f" at ({x}, {y})" if x is not None else "")
        facing = f", facing {e['direction']}" if e.get("direction") else ""
        user = (
            f"{where}{facing}, the step was refused and the game said: {sentence!r}.\n"
            "What gates this step, and what clears it? "
            'Respond with JSON {"gate": str, "clears_with": str}.'
        )
        row = chat(GATE_SYSTEM, user, json.dumps({"gate": cls, "clears_with": clears}), GATE_DOMAIN)
        row["meta"] = {
            "run_id": e.get("run_id"),
            "map": mp,
            "x": x,
            "y": y,
            "event": kind,
            "kind": e.get("kind"),
        }
        out.append(row)
    return out
