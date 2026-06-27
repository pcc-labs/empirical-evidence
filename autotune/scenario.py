"""The first-battle scenario: a constrained, keyless test that autotune can learn.

Route 1 is already solved by 80 hours of baked-in learnings, so its reward saturates and the
loop has nothing to improve. This scenario isolates a piece the genome *does* control: the
battle parameters. The game is reset to a captured first-battle save state, the agent fights
with the genome's battle params, and the reward is whether it won.

The experiment starts from deliberately-bad battle params (the "didn't know how to battle"
state) so there is a real gap to close, and we already know the good values, which gives clean
ground truth on whether the loop recovers them.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from autotune.genome import DEFAULT_PARAMS, PARAM_BOUNDS, clamp_params

# The battle-relevant knobs (the only params this scenario tunes).
BATTLE_PARAM_KEYS = (
    "hp_run_threshold",
    "hp_heal_threshold",
    "unknown_move_score",
    "status_move_score",
)

# Deliberately bad battle params: flee at high HP, waste turns healing, under-value real
# attacks, over-value zero-power status moves. This is the gap the loop has to close.
BATTLE_BAD = {
    "hp_run_threshold": 0.5,
    "hp_heal_threshold": 0.6,
    "unknown_move_score": 1.0,
    "status_move_score": 10.0,
}


def bad_battle_genome() -> dict:
    """Default genome with the battle params set to the deliberately-bad values."""
    return clamp_params({**DEFAULT_PARAMS, **BATTLE_BAD})


def battle_reward(fitness: dict) -> float:
    """Reward for one rollout: 1.0 if a battle was won, else 0.0."""
    return 1.0 if (fitness.get("battles_won", 0) or 0) >= 1 else 0.0


def mean_reward(fitnesses: list[dict], reward_fn: Callable[[dict], float]) -> float:
    """Mean reward over k rollouts of one genome (a finer, less noisy signal than one run)."""
    if not fitnesses:
        return 0.0
    return sum(reward_fn(f) for f in fitnesses) / len(fitnesses)


def _mutate_keys(genome: dict, keys: list[str], rng: random.Random, n: int) -> list[dict]:
    """Produce ``n`` candidates by perturbing 1-2 of ``keys`` within bounds. Other params held."""
    candidates: list[dict] = []
    for _ in range(n):
        cand = dict(genome)
        for key in rng.sample(keys, k=min(2, rng.choice([1, 2]))):
            lo, hi, typ = PARAM_BOUNDS[key]
            span = hi - lo
            step = rng.choice([-1, 1]) * span * rng.choice([0.15, 0.3, 0.5])
            cand[key] = typ(cand.get(key, DEFAULT_PARAMS[key]) + step)
        candidates.append(clamp_params(cand))
    return candidates


def mutate_battle_params(genome: dict, rng: random.Random, n: int) -> list[dict]:
    """Candidate genomes perturbing only battle params. Deterministic for a seeded ``rng``."""
    return _mutate_keys(genome, list(BATTLE_PARAM_KEYS), rng, n)


# ---------------------------------------------------------------------------
# Navigation scenario: get past the Route 1 choke (failure mode #3: exhaustion)
# ---------------------------------------------------------------------------

# The navigation / backtracking knobs that decide whether the agent gets unstuck or loops.
NAV_PARAM_KEYS = (
    "stuck_threshold",
    "door_cooldown",
    "waypoint_skip_distance",
    "bt_max_snapshots",
    "bt_restore_threshold",
    "bt_max_attempts",
    "bt_snapshot_interval",
)

# Deliberately bad navigation params: wait forever before skipping a waypoint, rarely skip,
# rarely backtrack, give up after one attempt. The "exhausted on an obstacle" starting point.
NAV_BAD = {
    "stuck_threshold": 20,
    "waypoint_skip_distance": 1,
    "bt_max_snapshots": 2,
    "bt_restore_threshold": 30,
    "bt_max_attempts": 1,
    "bt_snapshot_interval": 100,
    "door_cooldown": 16,
}


def bad_nav_genome() -> dict:
    """Default genome with navigation params set to the deliberately-bad values."""
    return clamp_params({**DEFAULT_PARAMS, **NAV_BAD})


def nav_reward(fitness: dict) -> float:
    """Progress reward: distinct maps reached from the start (1 = stuck, 2+ = escaped the choke)."""
    return float(fitness.get("maps_visited", 0) or 0)


def mutate_nav_params(genome: dict, rng: random.Random, n: int) -> list[dict]:
    """Candidate genomes perturbing only navigation params. Deterministic for a seeded ``rng``."""
    return _mutate_keys(genome, list(NAV_PARAM_KEYS), rng, n)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A constrained experiment: a starting genome, a mutation lens, and a reward."""

    name: str
    bad_genome: Callable[[], dict]
    mutate: Callable[[dict, random.Random, int], list[dict]]
    reward: Callable[[dict], float]
    battle_limit: int
    default_max_turns: int
    param_keys: tuple[str, ...]


SCENARIOS: dict[str, Scenario] = {
    "battle": Scenario(
        "battle", bad_battle_genome, mutate_battle_params, battle_reward, 1, 300, BATTLE_PARAM_KEYS
    ),
    "nav": Scenario("nav", bad_nav_genome, mutate_nav_params, nav_reward, 0, 800, NAV_PARAM_KEYS),
}
