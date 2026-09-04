"""Tetris placement corpus: runs/ -> neutral records -> SFT and eval views.

One row per graded decision. The board comes off the `placement_graded` event;
everything derivable from a board (the full ranking, a deeper grade) is
recomputed here, never stored. Prompts are rendered by tetris's own
`build_user_prompt` so training input is byte-identical to inference input.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

GRADE_KEYS = (
    "rank", "legal_count", "regret", "regret_norm", "best",
    "chosen_value", "best_value", "worst_value", "genome", "ply",
)


@dataclass(frozen=True)
class Record:
    run_id: str
    turn: int
    arm: str
    model: str
    harness: str | None
    effort: str | None
    seed: int
    mode: str
    board: list[str]
    piece: str
    next_piece: str
    chosen: list[int]
    reason: str
    grade: dict
    outcome: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_policy(policy: str) -> tuple[str, str | None, str | None]:
    """`pi/gemma4:26b/features/off` -> (model, harness, effort);

    `heuristic` -> (heuristic, None, None).
    """
    parts = policy.split("/")
    if parts[0] == "pi" and len(parts) >= 2:
        model, rest = "/".join(parts[:2]), parts[2:]
    else:
        model, rest = parts[0], parts[1:]
    harness = rest[0] if rest else None
    effort = rest[1] if len(rest) > 1 else None
    return model, harness, effort


def read_run(run_dir: Path) -> tuple[list[Record], Counter]:
    """Records for every graded decision that carries a board, plus counted exclusions."""
    run_dir = Path(run_dir)
    events = _events(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    fitness = summary.get("fitness", {})

    session = next(
        (
            e["data"]
            for e in events
            if e.get("event_type") == "session" and e.get("data", {}).get("phase") == "start"
        ),
        {},
    )
    policy = session.get("policy", "")
    p_model, p_harness, p_effort = _parse_policy(policy)
    model = session.get("model") or p_model
    harness = session.get("harness") or p_harness
    effort = session.get("effort") if "effort" in session else p_effort
    arm = session.get("arm") or meta.get("label") or policy
    seed = int(session.get("seed", session.get("timer_div", 0)))
    mode = session.get("mode", "paused")

    spawns = {e["turn"]: e["data"] for e in events if e.get("event_type") == "piece_spawn"}
    decisions: dict[int, dict] = {}
    for e in events:
        if e.get("event_type") == "placement_decision":
            decisions[e["turn"]] = e["data"]  # the last decision for a turn is the accepted one
    graded = {e["turn"]: e["data"] for e in events if e.get("event_type") == "placement_graded"}

    exclusions: Counter = Counter()
    for turn, d in decisions.items():
        if turn not in graded:
            exclusions["late" if d.get("late") else "ungraded"] += 1

    pieces_placed = int(fitness.get("pieces_placed", 0))
    outcome_base = {
        "final_score": int(fitness.get("score", 0)),
        "lines": int(fitness.get("lines", 0)),
        "pieces_placed": pieces_placed,
        "topped_out": bool(fitness.get("topped_out", False)),
    }

    records: list[Record] = []
    for turn in sorted(graded):
        g = graded[turn]
        if "board" not in g:
            exclusions["no_board"] += 1
            continue
        spawn = spawns.get(turn, {})
        records.append(
            Record(
                run_id=summary.get("run_id", run_dir.name),
                turn=turn,
                arm=arm,
                model=model,
                harness=harness,
                effort=effort,
                seed=seed,
                mode=mode,
                board=list(g["board"]),
                piece=spawn.get("piece", ""),
                next_piece=spawn.get("next_piece", ""),
                chosen=list(g["chosen"]),
                reason=decisions.get(turn, {}).get("reason", ""),
                grade={k: g[k] for k in GRADE_KEYS if k in g},
                outcome={**outcome_base, "pieces_after": pieces_placed - turn},
            )
        )
    return records, exclusions
