"""Try: run the pokemon-kafka agent N times and collect each rollout's telemetry.

Drives the agent through pokemon-kafka's already-wired ``EVOLVE_PARAMS`` seam (no pk edits),
exactly like ``pokemon-kafka/scripts/evolve.py::run_agent`` but with isolated per-rollout
telemetry so the verifier can read an ordered map sequence.

This module is an IO/subprocess wrapper and is exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from autotune.config import Config, load_config
from autotune.genome import base_genome
from autotune.story import load_story
from autotune.verifier import RolloutVerdict, load_fitness, load_game_events, verify

_ROLLOUT_TIMEOUT_S = 1200


@dataclass
class Rollout:
    """One agent run: the genome tried and the telemetry it produced."""

    index: int
    params: dict
    rollout_dir: Path
    fitness: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    returncode: int = 0
    timed_out: bool = False


def _agent_command(
    cfg: Config,
    rom: Path,
    fitness_path: Path,
    telemetry_dir: Path,
    max_turns: int,
    load_state: str | None = None,
    battle_limit: int = 0,
    save_state_on_trainer: str | None = None,
    worldmap_file: str | None = None,
    save_state_every: str | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(cfg.env.agent_script.resolve()),
        str(rom.resolve()),
        "--max-turns",
        str(max_turns),
        "--output-json",
        str(fitness_path.resolve()),
        "--telemetry-dir",
        str(telemetry_dir.resolve()),
        "--config",
        "",  # avoid depending on a local config.toml; JSONL telemetry still writes
    ]
    if battle_limit:
        cmd += ["--battle-limit", str(battle_limit)]
    if load_state:
        cmd += ["--load-state", str(Path(load_state).resolve())]
    if save_state_on_trainer:
        cmd += ["--save-state-on-trainer", save_state_on_trainer]
    if worldmap_file:
        cmd += ["--worldmap-file", str(Path(worldmap_file).resolve())]
    if save_state_every:
        cmd += ["--save-state-every", save_state_every]
    return cmd


def run_one(
    cfg: Config,
    params: dict,
    index: int,
    max_turns: int,
    work_root: Path,
    load_state: str | None = None,
    battle_limit: int = 0,
    save_state_on_trainer: str | None = None,
    worldmap_file: str | None = None,
    save_state_every: str | None = None,
) -> Rollout:
    """Run a single rollout to completion and load its artifacts."""
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM to run rollouts.")

    rollout_dir = work_root / f"rollout-{index}"
    telemetry_dir = rollout_dir / "telemetry"
    fitness_path = rollout_dir / "fitness.json"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Persist the genome alongside the fitness/telemetry so every rollout dir is self-describing
    # and minable later (the EVOLVE_PARAMS label is otherwise only in memory). Mirrors fitness.json.
    (rollout_dir / "genome.json").write_text(json.dumps(params))

    env = os.environ.copy()
    env["EVOLVE_PARAMS"] = json.dumps(params)

    cmd = _agent_command(
        cfg,
        cfg.env.rom_path,
        fitness_path,
        telemetry_dir,
        max_turns,
        load_state,
        battle_limit,
        save_state_on_trainer,
        worldmap_file,
        save_state_every,
    )
    rollout = Rollout(index=index, params=params, rollout_dir=rollout_dir)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.env.pokemon_kafka_dir.resolve()),
            env=env,
            capture_output=True,
            timeout=_ROLLOUT_TIMEOUT_S,
        )
        rollout.returncode = proc.returncode
    except subprocess.TimeoutExpired:
        rollout.timed_out = True

    rollout.fitness = load_fitness(fitness_path)
    rollout.events = load_game_events(telemetry_dir)
    return rollout


def run_batch(
    cfg: Config,
    params_list: list[dict],
    max_turns: int,
    work_root: Path,
    concurrency: int = 3,
    load_state: str | None = None,
    battle_limit: int = 0,
) -> list[Rollout]:
    """Run one rollout per genome in ``params_list``, in parallel. Order preserved."""
    work_root.mkdir(parents=True, exist_ok=True)
    results: list[Rollout | None] = [None] * len(params_list)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(run_one, cfg, params, i, max_turns, work_root, load_state, battle_limit): i
            for i, params in enumerate(params_list)
        }
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()
    return [r for r in results if r is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pokemon-kafka rollouts (the Try step).")
    parser.add_argument("--n", type=int, default=1, help="number of rollouts")
    parser.add_argument("--max-turns", type=int, default=1500)
    parser.add_argument("--work-root", type=str, default="./out/rollouts")
    args = parser.parse_args(argv)

    cfg = load_config()
    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    rollouts = run_batch(
        cfg,
        params_list=[base_genome() for _ in range(args.n)],
        max_turns=args.max_turns,
        work_root=Path(args.work_root),
        concurrency=cfg.loop.concurrency,
    )
    for r in rollouts:
        verdict: RolloutVerdict = verify(story, r.fitness, r.events)
        print(
            f"rollout {r.index}: reached '{verdict.furthest_beat_name}' "
            f"(beat {verdict.furthest_beat}/{story.target_beat_id}), "
            f"reward={verdict.story_reward}, on_story={verdict.on_story}, "
            f"rc={r.returncode}{' TIMEOUT' if r.timed_out else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
