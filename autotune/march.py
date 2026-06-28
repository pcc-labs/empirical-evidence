"""March to Pewter: run the agent in segments with a fresh turn budget each time, while
*persisting what it has learned* — the accumulated WorldMap (observations) and a checkpoint
save state — so progress compounds across runs instead of restarting blind every time.

This is the "reset turns while saving route.json and observations" loop:

  - **observations** — the agent's ``--worldmap-file`` (learned occupancy / walls / grass /
    encounter tiles) is loaded at the start of every segment and saved throughout, so the
    geometry it mapped in segment N is available in segment N+1.
  - **reset turns** — each segment runs a fresh ``segment_turns`` budget, resuming from the
    last checkpoint state (``--save-state-every``) rather than from the intro.
  - **route.json** — the ordered map sequence reached so far is written out each segment.

Stops as soon as the target map (Pewter City = 2) is reached, or after ``max_segments``.

IO/subprocess driver (like rollout/speedrun): exercised by live runs, not unit tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.genome import base_genome
from autotune.rollout import run_one
from autotune.verifier import extract_visited_maps

PEWTER_MAP_ID = 2


def _save_route(path: Path, visited: list[int], segment: int, final_map: int) -> None:
    """Persist the route discovered so far (ordered maps + furthest point)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"segment": segment, "final_map": final_map, "visited_maps": visited},
            indent=2,
        )
        + "\n"
    )


def run_march(
    cfg: Config,
    target_map: int = PEWTER_MAP_ID,
    segment_turns: int = 4000,
    max_segments: int = 15,
    checkpoint_every: int = 200,
    seed_state: str | None = None,
) -> dict:
    """Run segments until ``target_map`` is reached, persisting observations + a checkpoint."""
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM to run.")

    work_root = cfg.storage.out_dir / "march"
    wm_file = work_root / "worldmap.json"
    route_file = work_root / "route.json"
    checkpoint = (cfg.storage.out_dir / "march" / "checkpoint.state").resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    best_maps: list[int] = []
    for seg in range(max_segments):
        load_state = str(checkpoint) if checkpoint.exists() else seed_state
        r = run_one(
            cfg,
            base_genome(),
            seg,
            segment_turns,
            work_root / f"seg{seg}",
            load_state=load_state,
            worldmap_file=str(wm_file.resolve()),
            save_state_every=f"{checkpoint_every}:{checkpoint}",
        )
        visited = extract_visited_maps(r.events)
        final_map = r.fitness.get("final_map_id")
        if len(visited) > len(best_maps):
            best_maps = visited
        _save_route(route_file, visited, seg, final_map)
        reached = final_map == target_map or target_map in visited
        print(
            f"[march] seg {seg}: final_map={final_map} maps={visited} "
            f"battles_won={r.fitness.get('battles_won')} reached_target={reached} "
            f"rc={r.returncode}{' TIMEOUT' if r.timed_out else ''}",
            flush=True,
        )
        if reached:
            print(f"[march] reached target map {target_map} in segment {seg}.", flush=True)
            return {"reached": True, "segment": seg, "visited": visited, "worldmap": str(wm_file)}

    print(f"[march] did not reach map {target_map} in {max_segments} segments.", flush=True)
    return {
        "reached": False,
        "segments": max_segments,
        "visited": best_maps,
        "worldmap": str(wm_file),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="March to Pewter with persistent observations.")
    p.add_argument("--target-map", type=int, default=PEWTER_MAP_ID, help="map id to reach")
    p.add_argument("--segment-turns", type=int, default=4000, help="turn budget per segment")
    p.add_argument("--max-segments", type=int, default=15)
    p.add_argument("--seed-state", default=None, help="save state to start the first segment")
    p.add_argument("--fresh", action="store_true", help="discard any existing worldmap/checkpoint")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.fresh:
        for f in ("worldmap.json", "checkpoint.state", "route.json"):
            fp = cfg.storage.out_dir / "march" / f
            if fp.exists():
                fp.unlink()
    summary = run_march(
        cfg,
        target_map=args.target_map,
        segment_turns=args.segment_turns,
        max_segments=args.max_segments,
        seed_state=args.seed_state,
    )
    return 0 if summary["reached"] else 1


if __name__ == "__main__":
    sys.exit(main())
