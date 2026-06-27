# Constrained-test findings: can autotune learn by tuning the genome?

This documents three experiments run to test whether the autotune loop (Try, Check, Reward,
Nudge) can demonstrably *learn* by tuning the pokemon-kafka agent's 12-parameter genome. The
short answer is no, and the reason is consistent and instructive: the genome parameters have low
leverage over outcomes, so there is rarely a gap for the loop to close, and where a gap exists it
is hard to measure with the available signals.

The harness itself works. The walls are experimental-design walls, not code bugs.

## Experiment 1: Route 1 full run (`--nudge both`, 5 generations)

Ran the full loop from the game intro, story reward = per-beat progress toward Viridian City.

- Every rollout reached "Route 1" (beat 5 of 6) and no further. Reward was flat at 5.0 across all
  generations and all rollouts.
- The SFT training ran (validation loss fell 5.3 to 3.0), but on a signal-free dataset: every
  rollout scored the same, so "do more of what passed" had nothing to select on.

**Result: saturated.** Route 1 is already solved by 80 hours of baked-in learnings (hand-authored
`routes.json` waypoints, `EARLY_GAME_TARGETS`, battle heuristics), so there is no gap to improve.

## Experiment 2: first-battle savestate (battle params, win/lose reward)

Captured a save state at the first battle, started from deliberately-bad battle params, reward =
won the battle.

- Every rollout, every genome: `won=1`, `turns=14`, bit-for-bit identical.

**Result: saturated and deterministic.** Two compounding failures: (a) the first battle is
trivially winnable, so the battle params cannot make the agent lose it, and (b) a save state
replays deterministically (same RNG state), so every rollout plays the identical 14 turns
regardless of genome. Zero variance means no gradient for rejection sampling or RL.

## Experiment 3: Route 1 savestate (navigation params, progress reward)

Captured a save state on entering Route 1, started from deliberately-bad navigation params,
reward = distinct maps reached. Hill-climbed the nav params for 5 generations.

- Trajectory: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`. Completely flat.

Diagnostics revealed why:

| Genome | maps_visited | stuck_count | what it actually did |
|--------|--------------|-------------|----------------------|
| bad | 1 | 12 | sat in the grass grinding wild battles, never left Route 1 |
| default (good) | 3 | 92 | wandered **backward**: `12 -> 0 -> 37 -> 0 -> 12 -> ...` (Route 1 to Pallet to house, looping) |

- `maps_visited` is an invalid reward here: the "good" genome scored higher only by wandering the
  wrong direction (south, away from Viridian). `stuck_count` is backwards too (the good genome is
  *more* stuck because it thrashes). Neither genome reached Viridian.
- More turns did not help: 800, 1500, and 3000 turns all produced identical results.
- **Loading a save state restores the game state but not the agent's internal navigator memory**
  (waypoint index, recent positions, momentum). In a full run the agent reaches Route 1 with that
  state and continues north; cold-loaded, it has amnesia and loops. So save-state isolation, which
  worked conceptually, breaks the navigation agent.

**Result: invalid reward, broken-by-savestate agent, deterministic rollouts.**

## Root cause

All three point at the same thing: **the 12-parameter genome has low leverage over outcomes.** The
agent's behavior is driven by hardcoded routes, scripts, battle heuristics, and accumulated
internal state. The genome is a set of fine-tuning knobs, not a decision-maker. Tuning it rarely
flips success versus failure, so every constrained scenario either has no gap, or has a gap that
cannot be cleanly rewarded with the available fitness fields.

Three specific traps to remember:

1. **Saturation.** If the agent already succeeds (Route 1) or always succeeds (first battle), the
   reward is constant and there is nothing to learn.
2. **Determinism.** A save state replays identically, so `k > 1` rollouts add no information and
   the search has zero variance. Real RL needs stochastic rollouts.
3. **Reward validity.** `maps_visited` rewards backward wandering; `stuck_count` rewards standing
   still. A progress reward must be directional (did it get closer to the goal), and the available
   fitness fields do not express that.

## What the harness proved

The infrastructure is built and tested end to end, independent of the negative results:

- `loop.py` Try, Check, Reward, Nudge orchestration, with best-genome tracking.
- `rollout.py` runs the pk agent as subprocesses via `EVOLVE_PARAMS`, with `--load-state` and
  `--battle-limit` support.
- pokemon-kafka `agent.py`: `--load-state` (skip intro), `--save-state-on-battle`,
  `--save-state-on-map` (all at 100% coverage).
- `scenario.py` pluggable scenarios (battle, nav) with their own bad-start genome, mutation lens,
  and reward; `experiment.py` generic (1+lambda) hill-climb.
- The MLX path (rejection-sampling LoRA SFT, local-model proposer) runs locally and trains.

## What a learnable experiment would actually need

1. **A reward that measures directional progress toward the goal**, not breadth (`maps_visited`)
   or inactivity (`stuck_count`). For navigation this means "reached the target map/tile" or
   northward distance, which needs a fitness field the agent does not currently emit.
2. **A lever with real leverage.** Train the actual decision-maker (the policy proposing moves),
   not the 12 heuristic knobs, so that learning can flip outcomes.
3. **Rollout variance.** Save states replay deterministically; either run full stochastic rollouts
   or snapshot and restore the agent's internal state alongside the game state.

## Conclusion

The Try, Check, Reward, Nudge loop and its pokemon-kafka integration are built and verified. But
demonstrating learning needs a task where the thing being tuned controls the outcome and the
reward cleanly measures progress. Tuning the 12-parameter genome on early-game Pokemon is not that
task. The next meaningful step, if pursued, is to move the lever from the heuristic genome to the
decision policy, and to define a directional progress reward.
