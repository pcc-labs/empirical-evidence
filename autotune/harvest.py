"""Harvest a real, labeled SFT corpus by running diverse genomes from captured save states.

The in-loop SFT buffer only sees the few winners of a single loop run (16 examples on Jun 27 —
a proof-of-life, not a learned policy). This module is the data *engine*: it loads a spread of
captured states (``states/route1.state`` etc.), runs a diverse genome population from each, scores
every rollout with the verifier, and rejection-samples improvement pairs into ``data/sft``.

Seeding from a state concentrates data at the wall (every rollout starts at the frontier instead
of re-traversing solved early game) and is deterministic per ``(state, genome)`` — so k=1, no
wasted replication. The genome is the *label* the proposer must predict, so unlike mining old
rollout dirs (which never persisted the genome), here we choose and record it.

Pure seams (``build_genome_population``, ``assemble_corpus``, ``resolve_states``) are unit-tested;
the subprocess driver and CLI are exercised by the smoke run.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.genome import PARAM_BOUNDS, base_genome, clamp_params
from autotune.nudge_sft import Winner, assemble_corpus, write_corpus, write_sft_data
from autotune.rollout import run_batch
from autotune.scenario import NAV_PARAM_KEYS, mutate_nav_params
from autotune.story import load_story
from autotune.verifier import verify


def _random_nav_genome(rng: random.Random) -> dict:
    """A genome with all navigation params drawn uniformly within bounds (battle params held)."""
    g = base_genome()
    for key in NAV_PARAM_KEYS:
        lo, hi, _typ = PARAM_BOUNDS[key]
        g[key] = rng.randint(int(lo), int(hi))
    return clamp_params(g)


def build_genome_population(seed: int, n: int) -> list[dict]:
    """A diverse, deterministic population of ``n`` nav genomes.

    Always includes the default genome as a reference point, then mixes wide random nav samples
    with modest mutations of the default so the harvest spans both exploration and exploitation.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pop = [base_genome()]
    while len(pop) < n:
        if rng.random() < 0.5:
            pop.append(_random_nav_genome(rng))
        else:
            pop.append(mutate_nav_params(base_genome(), rng, 1)[0])
    return pop[:n]


def resolve_states(spec: str) -> list[str]:
    """Resolve a ``--states`` spec to paths: a dir -> all its ``*.state``; a file -> just itself."""
    path = Path(spec)
    if path.is_dir():
        return [str(p.resolve()) for p in sorted(path.glob("*.state"))]
    if path.is_file():
        return [str(path.resolve())]
    return []


def run_harvest(  # pragma: no cover - subprocess driver, exercised by the smoke run
    cfg: Config,
    state_paths: list[str],
    n_genomes: int,
    max_turns: int,
    seed: int,
) -> dict:
    """Run the population from each state, score, and write the rejection-sampled SFT corpus."""
    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    population = build_genome_population(seed, n_genomes)
    winners_by_state: dict[str, list[Winner]] = {}
    for state in state_paths:
        work_root = cfg.storage.out_dir / "harvest" / Path(state).stem
        rollouts = run_batch(
            cfg,
            params_list=[dict(g) for g in population],
            max_turns=max_turns,
            work_root=work_root,
            concurrency=cfg.loop.concurrency,
            load_state=state,
        )
        winners_by_state[state] = [
            Winner(r.params, verify(story, r.fitness, r.events)) for r in rollouts
        ]
        on_story = sum(w.verdict.on_story for w in winners_by_state[state])
        print(f"[harvest] {Path(state).name}: {len(rollouts)} rollouts, {on_story} on-story")

    examples = assemble_corpus(winners_by_state, story)
    corpus_path = write_corpus(cfg.storage.sft_dir / "corpus.jsonl", examples)
    train_path, valid_path = write_sft_data(cfg.storage.sft_dir, examples, seed=seed)
    return {
        "states": len(state_paths),
        "genomes": len(population),
        "examples": len(examples),
        "corpus_path": str(corpus_path),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    p = argparse.ArgumentParser(description="Harvest a labeled SFT corpus from save states.")
    p.add_argument("--genomes", type=int, default=24, help="genome population size per state")
    p.add_argument("--states", default="states/", help="a .state file or a dir of them")
    p.add_argument("--max-turns", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = load_config()
    state_paths = resolve_states(args.states)
    if not state_paths:
        print(f"No save states found at {args.states!r}. Capture one first.", file=sys.stderr)
        return 1

    summary = run_harvest(cfg, state_paths, args.genomes, args.max_turns, args.seed)
    print(
        f"[harvest] done: {summary['examples']} examples from {summary['genomes']} genomes "
        f"x {summary['states']} states -> {summary['train_path']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
