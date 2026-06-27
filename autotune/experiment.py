"""Keyless constrained experiment: can the loop tune a genome from a bad start?

A generic (1+lambda) hill-climb over a scenario's parameters, evaluated against a captured
save state. No API key and no MLX. Scenarios live in autotune.scenario:

  - battle: tune battle params against a captured first-battle state (reward = win).
  - nav:    tune navigation params against a captured Route 1 state (reward = maps reached).

The output is a reward trajectory. If it climbs from the bad start and recovers sensible
params, the loop learns. If it stays flat, it does not (saturated reward or no variance).

IO/subprocess driver, exercised by the experiment run, not unit tests.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.rollout import run_batch
from autotune.scenario import SCENARIOS, Scenario, mean_reward


def _params(genome: dict, keys: tuple[str, ...]) -> dict:
    return {k: round(float(genome[k]), 3) for k in keys}


def _evaluate(
    cfg: Config,
    scenario: Scenario,
    genome: dict,
    k: int,
    max_turns: int,
    state_path: str,
    label: str,
) -> float:
    """Mean reward of one genome over k rollouts from the save state."""
    work_root = cfg.storage.out_dir / scenario.name / label
    rollouts = run_batch(
        cfg,
        params_list=[dict(genome) for _ in range(k)],
        max_turns=max_turns,
        work_root=work_root,
        concurrency=cfg.loop.concurrency,
        load_state=state_path,
        battle_limit=scenario.battle_limit,
    )
    return mean_reward([r.fitness for r in rollouts], scenario.reward)


def run_experiment(
    cfg: Config,
    scenario: Scenario,
    state_path: str,
    generations: int,
    population: int,
    k_rollouts: int,
    max_turns: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    best = scenario.bad_genome()
    best_reward = _evaluate(cfg, scenario, best, k_rollouts, max_turns, state_path, "seed")
    print(
        f"[{scenario.name}] start (bad params): reward={best_reward:.2f} "
        f"{_params(best, scenario.param_keys)}"
    )
    history = [best_reward]

    for gen in range(1, generations + 1):
        scored = []
        for ci, cand in enumerate(scenario.mutate(best, rng, population)):
            r = _evaluate(cfg, scenario, cand, k_rollouts, max_turns, state_path, f"gen{gen}-c{ci}")
            scored.append((r, cand))
        gen_reward, gen_best = max(scored, key=lambda t: t[0])
        if gen_reward > best_reward:
            best_reward, best = gen_reward, gen_best
        history.append(best_reward)
        print(
            f"[{scenario.name}] gen {gen}: best reward={best_reward:.2f} "
            f"candidates={[round(r, 2) for r, _ in scored]} {_params(best, scenario.param_keys)}"
        )

    print(f"[{scenario.name}] trajectory: {[round(h, 2) for h in history]}")
    print(f"[{scenario.name}] final params: {_params(best, scenario.param_keys)}")
    return {"history": history, "best": best, "best_reward": best_reward}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Keyless constrained tuning experiment.")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="nav")
    p.add_argument("--state", required=True, help="captured save state path")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--population", type=int, default=4, help="candidate genomes per generation")
    p.add_argument("--k", type=int, default=3, help="rollouts per candidate (reward samples)")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = load_config()
    scenario = SCENARIOS[args.scenario]
    max_turns = args.max_turns if args.max_turns is not None else scenario.default_max_turns
    state = str(Path(args.state).resolve())
    if not Path(state).exists():
        print(f"No save state at {state}. Capture one first (scripts/experiment.sh).",
              file=sys.stderr)
        return 1
    run_experiment(
        cfg, scenario, state, args.generations, args.population, args.k, max_turns, args.seed
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
