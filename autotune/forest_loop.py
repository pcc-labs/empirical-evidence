"""Forest training loop: evolve nav params to climb the dense forest sub-beat reward.

Mirrors :mod:`autotune.brock_loop`, but the lever is NAVIGATION (the forest is a maze, not a
battle) and the reward is the ordered sub-beat ladder from :mod:`autotune.forest_story`. Each
rollout starts from a genuinely-leveled forest-entrance state (party.py with the EXP fix, so the
lead doesn't collapse to L7 and white out) and is scored by how many forest sub-beats it reaches
IN ORDER. Per generation we mutate the nav genome, keep the best, and record a leaderboard.

This is the param-evolution form of training — runnable with no model download. Its winners are
exactly the high-reward rollouts the LoRA path (``nudge_sft`` + ``train_sft``) trains on next.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from autotune.config import Config
from autotune.forest_story import FOREST_BEATS, score_forest
from autotune.genome import base_genome
from autotune.rollout import run_batch
from autotune.scenario import mutate_nav_params

DEFAULT_FOREST_STATE = "./states/forest_lv/lead_lv13.state"


def _evaluate(cfg: Config, genome: dict, state: str, label: str, max_turns: int):
    """One rollout of ``genome`` from the forest-entrance state; return (verdict, dir, fitness)."""
    work_root = cfg.storage.out_dir / "forest" / label
    rollouts = run_batch(
        cfg,
        params_list=[dict(genome)],
        max_turns=max_turns,
        work_root=work_root,
        concurrency=1,
        load_state=state,
    )
    r = rollouts[0]
    return score_forest(r.events), str(r.rollout_dir), r.fitness


def _key(verdict, fitness: dict) -> tuple:
    """Rank by sub-beats reached, then fewer turns (a faster route to the same beat is better)."""
    return (verdict.reward, -float(fitness.get("turns", 1e9)))


def _entry(verdict, genome: dict, rollout_dir: str, fitness: dict) -> dict:
    return {
        "reward": verdict.reward,
        "furthest_beat": verdict.furthest_beat,
        "furthest_beat_name": verdict.furthest_beat_name,
        "per_beat": list(verdict.per_beat),
        "crossed": verdict.crossed,
        "trainer_wins": verdict.signals.trainer_wins,
        "sign_read": verdict.signals.sign_read,
        "turns": fitness.get("turns"),
        "rollout_dir": rollout_dir,
    }


def run_forest_loop(
    cfg: Config,
    state: str = DEFAULT_FOREST_STATE,
    generations: int = 6,
    population: int = 4,
    max_turns: int = 1500,
    seed: int = 42,
) -> dict:
    """Evolve nav params to climb the forest sub-beat reward. Returns a summary."""
    state = str(Path(state).resolve())
    if not Path(state).exists():
        raise RuntimeError(
            f"No forest-entrance state at {state}. Poke a genuinely-leveled one first:\n"
            "  python -m autotune.party $ROM_PATH --in-state states/forest_healed.state "
            "--levels 13 --out-dir states/forest_lv"
        )

    rng = random.Random(seed)
    genome = base_genome()
    entries: list[dict] = []

    best, rdir, best_fit = _evaluate(cfg, genome, state, "seed", max_turns)
    best_genome = dict(genome)
    entries.append(_entry(best, genome, rdir, best_fit))
    print(
        f"[forest] seed: reward={best.reward:.0f}/{len(FOREST_BEATS)} "
        f"beat='{best.furthest_beat_name}' crossed={best.crossed}"
    )

    for gen in range(generations):
        scored = []
        for ci, cand in enumerate(mutate_nav_params(genome, rng, population)):
            v, rdir, fit = _evaluate(cfg, cand, state, f"gen{gen}-c{ci}", max_turns)
            scored.append((v, cand, fit))
            entries.append(_entry(v, cand, rdir, fit))

        gv, gg, gfit = max(scored, key=lambda t: _key(t[0], t[2]))
        if _key(gv, gfit) > _key(best, best_fit):
            best, genome, best_fit, best_genome = gv, gg, gfit, dict(gg)
        else:
            genome = best_genome  # keep mutating from the best-so-far

        print(
            f"[forest] gen {gen}: best_reward={best.reward:.0f}/{len(FOREST_BEATS)} "
            f"beat='{best.furthest_beat_name}' crossed={best.crossed} "
            f"cands={[v.reward for v, _, _ in scored]}"
        )
        if best.crossed:
            print("[forest] crossed the forest — stopping.")
            break

    _write_outputs(cfg, entries, best, best_genome, state)
    return {
        "best_reward": best.reward,
        "crossed": best.crossed,
        "furthest": best.furthest_beat_name,
        "evaluations": len(entries),
    }


def _write_outputs(cfg: Config, entries, best, best_genome: dict, state: str) -> None:
    out_dir = cfg.storage.out_dir / "forest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(entries, key=lambda e: (e["reward"], -(e.get("turns") or 1e9)), reverse=True)
    (out_dir / "leaderboard.json").write_text(
        json.dumps({"task": "forest_crossing", "best": ranked[0] if ranked else None,
                    "entries": ranked}, indent=2) + "\n"
    )
    (cfg.storage.out_dir / "best_forest.json").write_text(
        json.dumps({"genome": best_genome, "state": state, "reward": best.reward,
                    "furthest_beat": best.furthest_beat_name, "crossed": best.crossed},
                   indent=2) + "\n"
    )
    print(f"[forest] leaderboard -> {out_dir / 'leaderboard.json'}")
    print(f"[forest] best config -> {cfg.storage.out_dir / 'best_forest.json'}")
