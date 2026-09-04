"""Tetris placement corpus: runs/ -> neutral records -> SFT and eval views.

One row per graded decision. The board comes off the `placement_graded` event;
everything derivable from a board (the full ranking, a deeper grade) is
recomputed here, never stored. Prompts are rendered by tetris's own
`build_user_prompt` so training input is byte-identical to inference input.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tetris_agent.quality import TOP_OUT_VALUE  # one source of truth for the lethal sentinel

GRADE_KEYS = (
    "rank", "legal_count", "regret", "regret_norm", "best",
    "chosen_value", "best_value", "worst_value", "genome", "ply",
)

DEATH_SPIRAL = 5
TRAIN_SEED_MIN = 100
EVAL_SEEDS = (1, 2, 3, 4, 5)
# The served live prompt's level-0 ceiling: tetris_agent.live_agent._deadline_s is
# max(1.0, rows_to_fall * frames_per_row / 60 - _EXEC_HEADROOM_S), and at level 0
# with Emulator._GRAVITY_RELOADS[0] == 52, ROWS == 18, _EXEC_HEADROOM_S == 2.0, the
# maximum (a piece spawning at the very top, rows_to_fall == ROWS - 1 == 17) is
# 17 * 53 / 60 - 2.0 = 13.017s. 13.0 stays under that ceiling for every board depth
# and every level >= 0 the served prompt can produce (a real spawn is usually ~12s).
LIVE_DEADLINE_S = 13.0


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


def apply_filters(
    records: list[Record], death_spiral: int = DEATH_SPIRAL
) -> tuple[list[Record], Counter]:
    """Outcome-based filters, plus the one oracle veto it is safe to make.

    The grader (~490-level) is weaker than the teacher (530), so regret and rank
    are never filters. The spiral rule drops the last `death_spiral` decisions of
    a run that topped out; the veto drops a move that left the next piece nowhere
    to go when a survivable move existed.
    """
    dropped: Counter = Counter()
    by_run: dict[str, list[Record]] = {}
    for r in records:
        by_run.setdefault(r.run_id, []).append(r)

    kept: list[Record] = []
    for run_records in by_run.values():
        run_records = sorted(run_records, key=lambda r: r.turn)
        if run_records and run_records[0].outcome.get("topped_out"):
            cut = max(0, len(run_records) - death_spiral)
            dropped["death_spiral"] += len(run_records) - cut
            run_records = run_records[:cut]
        for r in run_records:
            chosen_value = r.grade.get("chosen_value", 0.0)
            best_value = r.grade.get("best_value", 0.0)
            if chosen_value <= TOP_OUT_VALUE and best_value > TOP_OUT_VALUE:
                dropped["top_out_veto"] += 1
                continue
            kept.append(r)
    return kept, dropped


def split(records: list[Record]) -> tuple[list[Record], list[Record], Counter]:
    """Train rows come only from seeds >= TRAIN_SEED_MIN; validation rows only from EVAL_SEEDS."""
    train: list[Record] = []
    valid: list[Record] = []
    excluded: Counter = Counter()
    for r in records:
        if r.seed >= TRAIN_SEED_MIN:
            train.append(r)
        elif r.seed in EVAL_SEEDS:
            valid.append(r)
        else:
            excluded["seed_out_of_pool"] += 1
    return train, valid, excluded


def board_array(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def sft_row(record: Record) -> dict:
    """`{"messages": [system, user, assistant]}` — the format train_sft.py reads.

    Rendered through tetris's own prompt code and the pi arm's exact suffixes, so
    the training input is byte-identical to what the served student receives.
    The deadline line is the level-0 fall time even for rows minted paused: the
    deployed prompt is the live one.
    """
    from tetris_agent.pi_policy import PI_JSON_INSTRUCTIONS, PI_PROMPT_SUFFIX
    from tetris_agent.prompts import build_user_prompt, legal_placements, system_prompt_for

    harness = record.harness or "features"
    board = board_array(record.board)
    placements = legal_placements(board, record.piece)
    user = build_user_prompt(
        harness, board, record.piece, record.next_piece, placements, record.turn,
        deadline_s=LIVE_DEADLINE_S,
    )
    assistant = json.dumps(
        {"rotation": record.chosen[0], "col": record.chosen[1], "reason": record.reason}
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt_for(harness) + PI_JSON_INSTRUCTIONS},
            {"role": "user", "content": user + PI_PROMPT_SUFFIX},
            {"role": "assistant", "content": assistant},
        ]
    }


GENOME_KEYS = ("w_lines", "w_agg_height", "w_holes", "w_bumpiness")


def eval_row(record: Record, ply: int = 2) -> dict:
    """A held-out row for tier-1: the prompt, the teacher's choice, and the oracle's
    full ranking recomputed from the board — so the training job grades answers
    with no tetris dependency of its own."""
    from tetris_agent.policy import Genome
    from tetris_agent.quality import rank_placements

    genome = Genome(**{k: record.grade["genome"][k] for k in GENOME_KEYS})
    ranked = rank_placements(
        board_array(record.board), record.piece, record.next_piece, genome, ply
    )
    messages = sft_row(record)["messages"][:2]
    return {
        "run_id": record.run_id,
        "seed": record.seed,
        "turn": record.turn,
        "messages": messages,
        "teacher": list(record.chosen),
        "ranking": [[rot, col, round(value, 6)] for (rot, col), value in ranked],
    }


def corpus_id(records: list[Record]) -> str:
    """`YYYYMMDD-<12 hex>`: the date plus a hash of the record set, order-independent."""
    keys = sorted(f"{r.run_id}:{r.turn}:{r.chosen}" for r in records)
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
    return f"{datetime.now(timezone.utc):%Y%m%d}-{digest}"


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def build_corpus(runs_dir: Path, out_root: Path, ply: int = 2) -> Path:
    """Read every run under runs_dir into <out_root>/<corpus_id>/ and return that directory."""
    runs_dir, out_root = Path(runs_dir), Path(out_root)
    all_records: list[Record] = []
    excluded: Counter = Counter()
    n_runs = 0
    for run_dir in sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file()
    ):
        n_runs += 1
        records, ex = read_run(run_dir)
        all_records.extend(records)
        excluded.update(ex)

    kept, dropped = apply_filters(all_records)
    excluded.update(dropped)
    train, valid, stray = split(kept)
    excluded.update(stray)

    out = out_root / corpus_id(kept)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "records.jsonl", (r.to_dict() for r in kept))
    n_train = _write_jsonl(out / "train.jsonl", (sft_row(r) for r in train))
    n_valid = _write_jsonl(out / "valid.jsonl", (sft_row(r) for r in valid))
    _write_jsonl(out / "eval.jsonl", (eval_row(r, ply) for r in valid))

    per_seed: Counter = Counter(str(r.seed) for r in kept)
    regrets = [r.grade.get("regret_norm", 0.0) for r in kept]
    top1 = [1.0 if r.grade.get("rank") == 1 else 0.0 for r in kept]
    stats = {
        "corpus_id": out.name,
        "runs": n_runs,
        "records": len(all_records),
        "kept": len(kept),
        "excluded": dict(excluded),
        "per_seed": dict(sorted(per_seed.items())),
        "train_rows": n_train,
        "valid_rows": n_valid,
        "ply": ply,
        "teacher": {
            "mean_regret_norm": round(sum(regrets) / len(regrets), 6) if regrets else None,
            "top1_rate": round(sum(top1) / len(top1), 6) if top1 else None,
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tetris_corpus", description="runs/ -> placement corpus")
    ap.add_argument("--runs", default="../tetris/runs")
    ap.add_argument("--out", default="data/tetris")
    ap.add_argument("--ply", type=int, default=2)
    ap.add_argument(
        "--hf-repo", default="bdougie/tetris-placements", help="dataset repo the upload hint names"
    )
    args = ap.parse_args(argv)
    out = build_corpus(Path(args.runs), Path(args.out), ply=args.ply)
    stats = json.loads((out / "stats.json").read_text())
    print(json.dumps(stats, indent=2))
    print(f"\nwritten: {out}")
    print(f"upload:  hf upload {args.hf_repo} {out} {out.name} --type dataset --private")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
