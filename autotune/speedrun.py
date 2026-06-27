"""Phase A: speedrun to Pewter, study the path, capture a pre-Brock save state.

Runs one full agent rollout toward Pewter City and turns its telemetry into a **PathStudy**:
the ordered maps with turns-per-segment, the total turns, and the Brock-fight summary. The same
run dumps a save state the instant Brock's fight begins (``--save-state-on-trainer brock:...``)
for Phase B to optimize against.

``build_path_study`` is pure and unit-tested; ``run_speedrun`` is the subprocess driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from autotune.brock import _find_brock_end, extract_brock_fitness
from autotune.config import load_config
from autotune.story import _FALLBACK_NAMES, derive_story_from_run, load_routes
from autotune.verifier import extract_visited_maps, load_fitness, load_game_events


@dataclass(frozen=True)
class Segment:
    """One contiguous stay on a map."""

    map_id: int
    name: str
    entered_turn: int
    turns: int


@dataclass(frozen=True)
class BrockFight:
    """Summary of the Brock fight as it played out in the speedrun."""

    reached: bool
    won: bool | None
    turns: int | None
    lead_species: str | None
    lead_level: int | None
    gym_map_id: int | None


@dataclass(frozen=True)
class PathStudy:
    """What the run did: trajectory, turns-per-segment, total turns, and the Brock outcome."""

    reached_pewter: bool
    total_turns: int
    visited_maps: tuple[int, ...]
    segments: tuple[Segment, ...]
    brock: BrockFight


def _map_timeline(events: list[dict]) -> list[tuple[int, int]]:
    """Ordered ``(turn, map_id)`` points from overworld + map_change events."""
    pts: list[tuple[int, int]] = []
    for ev in sorted(events, key=lambda e: e.get("turn", 0)):
        data = ev.get("data", {})
        etype = ev.get("event_type")
        if etype == "overworld":
            map_id = data.get("map_id")
        elif etype == "map_change":
            map_id = data.get("new_map")
        else:
            continue
        if isinstance(map_id, int):
            pts.append((ev.get("turn", 0), map_id))
    return pts


def _segments(events: list[dict], end_turn: int, routes: dict[int, dict]) -> list[Segment]:
    """Collapse the map timeline into contiguous segments with turns-per-segment."""
    timeline = _map_timeline(events)
    starts: list[tuple[int, int]] = []  # (map_id, entered_turn)
    for turn, map_id in timeline:
        if starts and starts[-1][0] == map_id:
            continue
        starts.append((map_id, turn))

    segments: list[Segment] = []
    for i, (map_id, start) in enumerate(starts):
        next_start = starts[i + 1][1] if i + 1 < len(starts) else end_turn
        name = (routes.get(map_id, {}) or {}).get("name") or _FALLBACK_NAMES.get(
            map_id, f"Map {map_id}"
        )
        segments.append(
            Segment(map_id=map_id, name=name, entered_turn=start, turns=max(0, next_start - start))
        )
    return segments


def build_path_study(
    events: list[dict], fitness: dict, routes: dict[int, dict] | None = None
) -> PathStudy:
    """Turn a run's telemetry + fitness into a PathStudy. Pure."""
    routes = routes or {}
    visited = extract_visited_maps(events)
    total_turns = int(fitness.get("turns", 0) or 0)
    if total_turns == 0 and events:
        total_turns = max(e.get("turn", 0) for e in events)

    # Story derived from the observed path (target = Pewter if reached, else the furthest map).
    story = derive_story_from_run(visited, routes=routes, target_map_id=2 if 2 in visited else None)
    metrics = extract_brock_fitness(events, fitness, story)

    brock_end = _find_brock_end(events, brock_map_id=None)
    gym_map_id = brock_end.get("data", {}).get("map_id") if brock_end else None

    brock = BrockFight(
        reached=metrics["reached_pewter"],
        won=metrics["won"] if (metrics["turns"] is not None or brock_end) else None,
        turns=metrics["turns"],
        lead_species=metrics["lead_species"],
        lead_level=metrics["lead_level"],
        gym_map_id=gym_map_id,
    )
    return PathStudy(
        reached_pewter=2 in visited,
        total_turns=total_turns,
        visited_maps=tuple(visited),
        segments=tuple(_segments(events, total_turns, routes)),
        brock=brock,
    )


def study_to_dict(study: PathStudy) -> dict:
    """JSON-serializable view of a PathStudy."""
    return asdict(study)


def run_speedrun(
    cfg,
    max_turns: int = 6000,
    pre_brock_state: str = "./states/pre_brock.state",
    out_path: str = "./out/path_study.json",
) -> PathStudy:
    """Run one full rollout to Pewter, capture the pre-Brock state, and write the PathStudy."""
    from autotune.genome import base_genome
    from autotune.rollout import run_one

    state_path = Path(pre_brock_state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = cfg.storage.out_dir / "speedrun"
    work_root.mkdir(parents=True, exist_ok=True)

    rollout = run_one(
        cfg,
        params=base_genome(),
        index=0,
        max_turns=max_turns,
        work_root=work_root,
        battle_limit=0,
        # Dump a state the instant Brock's fight begins (first gym-leader-level trainer).
        save_state_on_trainer=f"brock:{state_path.resolve()}",
    )

    routes = load_routes(cfg.env.routes_json)
    study = build_path_study(rollout.events, rollout.fitness, routes)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(study_to_dict(study), indent=2) + "\n")
    return study


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Speedrun to Pewter + study the path (Phase A).")
    p.add_argument("--max-turns", type=int, default=6000)
    p.add_argument("--pre-brock-state", default="./states/pre_brock.state")
    p.add_argument("--out", default="./out/path_study.json")
    p.add_argument(
        "--from-telemetry",
        default=None,
        help="Skip running: build the study from an existing telemetry dir (the one containing "
        "game/), reading fitness.json beside it if present",
    )
    args = p.parse_args(argv)
    cfg = load_config()

    if args.from_telemetry:
        tdir = Path(args.from_telemetry)
        events = load_game_events(tdir)  # reads <tdir>/game/*.jsonl
        fitness = load_fitness(tdir / "fitness.json")
        study = build_path_study(events, fitness, load_routes(cfg.env.routes_json))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(study_to_dict(study), indent=2) + "\n")
    else:
        study = run_speedrun(cfg, args.max_turns, args.pre_brock_state, args.out)

    print(
        f"[speedrun] reached_pewter={study.reached_pewter} total_turns={study.total_turns} "
        f"segments={len(study.segments)} brock_won={study.brock.won} "
        f"brock_turns={study.brock.turns}"
    )
    print(f"[speedrun] path study -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
