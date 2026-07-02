# Constrained-test findings: can empirical-evidence learn by tuning the genome?

This documents three experiments run to test whether the empirical-evidence loop (Try, Check, Reward,
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

## 2026-07-02 domain-tagged corpus smoke

End-to-end smoke of the domain-tagged corpus pipeline (harvest -> merge -> train_sft -> per-domain
benchmark) on the cuda box (RTX 5090). Three findings plus the smoke evidence:

1. **Canonical-route crossings are genome-invariant at lv13.** Every survival genome swept over
   the canonical forest route crossed identically (reward 3.0, 154 turns, full HP throughout), so
   the harvest had no gradient. Survival pairs must be harvested in legacy BEATS nav
   (`forest_harvest --route ''`, passthrough landed as `fda7d6f`), where genomes produce real
   spread (turns 310-414; one flee-late genome fainted at 0/37 HP). The canonical route remains
   the right default for the benchmark itself.
2. **Nav-tagged pairs are unobtainable with today's agent+states.** pokemon-kafka's agent.py does
   consume `EVOLVE_PARAMS`, but deliberately pathological nav params (`stuck_threshold: 20`,
   `waypoint_skip_distance: 1`, `bt_max_attempts: 1`, ...) vs the default genome from
   `route2.state` produce byte-identical verdicts at both 400 and 1200 turns (story_reward=6.0,
   score=8515.000@400t / 8435.000@1200t, final_map=51): nav params only bifurcate behavior when
   the stuck/backtrack/door branches fire, and the agent navigates cleanly. Blocked until state
   capture or pk-side work (both out of scope) — parallel to the accepted bag_count gap for
   discovery beats.
3. **The smoke caught a real OOM:** benchmarking 4 checkpoints in one process accumulated a full
   base-model copy per checkpoint (~30 GiB) via generate.py's model cache; fixed by evicting the
   resident model on adapter switch + checkpoint-outer sweep (`4d973b1`).

Merge census: `{"battle": 5}` (5 pairs from a 6-genome BEATS sweep; no nav or discovery rows).

```
forest benchmark: proposer genome per checkpoint vs base genome (baseline reward 3.00)
  checkpoint   reward    nav  battle  discov  crossed
    baseline     3.00   2.00    1.00    0.00     1.00
          10     3.00   2.00    1.00    0.00     1.00
          20     3.00   2.00    1.00    0.00     1.00
          30     3.00   2.00    1.00    0.00     1.00
          40     3.00   2.00    1.00    0.00     1.00
       final     3.00   2.00    1.00    0.00     1.00
```

Caveat: 5 battle-only pairs at 40 iters is a mechanical smoke of the pipeline (train loss ~0.002,
memorization of 4 rows), not a learned policy — the flat trend table is expected and must not be
read as signal. Artifacts: `out/forest_benchmark_trends.png`, `out/lora_weight_trends.png`.

### Gradient run (lv6 benchmark state)

Followed the smoke with a bigger, two-state BEATS harvest (`--route ''`, lv6 + lv13 leads, both
starting map 51 pos (5,0)) and a benchmark on the lv6 state, which the smoke could not use because
lv13 is unkillable there.

**Harvest — lv6 lead** (`states/forest_healed.state`, 21/21 HP, 8-genome sweep): 4 SFT pairs from 8
runs. Reward was flat at 1.0 for all 8 genomes and none crossed, but raw survival (turns before the
rollout ended) spread with `hp_run_threshold`: 108 turns at 0.1/0.25, 108 at 0.25, 133 at 0.4, 138
at 0.6 (heal threshold 0.25 vs 0.5 made no difference at this level).

**Harvest — lv13 lead** (`states/forest_lv/lead_lv13_potions.state`, 34/34 HP, 8-genome sweep): 7
SFT pairs from 8 runs. Reward again flat at 1.0 for all genomes, none crossed, but turns spread much
wider: 310, 319, 310, 319, 112, 301, 359, 414. The `hp_run_threshold=0.6 / hp_heal_threshold=0.5`
genome (turns=414, the longest-surviving) fainted at 0/37 HP in the raw log — a genuine survival
failure at the far end of the sweep, matching the "flee-late genome faints" pattern noted in finding
1 above. The `hp_run_threshold=0.4 / hp_heal_threshold=0.25` genome is a low outlier (112 turns);
raw log shows it lost the fight early rather than surviving longer.

**Merge**: `data/sft_union2`, 11 examples, census `{"battle": 11}` — more than double the smoke's 5
pairs, still battle-only (no nav or discovery rows, consistent with finding 2 above).

**Training**: `out/sft`, 60 iters / save-steps 15, checkpoints 15/30/45/60 all present. Train loss
converged to ~0.003 (mean_token_accuracy 0.998) — with only 11 examples this is memorization, not
generalization, same caveat as the smoke.

**Benchmark** (lv6 state, route-mode default, baseline reward 2.00):

```
forest benchmark: proposer genome per checkpoint vs base genome (baseline reward 2.00)
  checkpoint   reward    nav  battle  discov  crossed
    baseline     2.00   1.00    1.00    0.00     0.00
          15     2.00   1.00    1.00    0.00     0.00
          30     2.00   1.00    1.00    0.00     0.00
          45     2.00   1.00    1.00    0.00     0.00
          60     2.00   1.00    1.00    0.00     0.00
       final     2.00   1.00    1.00    0.00     0.00
```

**What this does and doesn't show.** The harvest sweeps *do* have real gradient this time — turns
before ending the rollout vary 3-4x across genomes at both levels, and the lv13 sweep even shows a
genuine faint at the survival-maximizing extreme of the sweep. But the benchmark trend is still
completely flat, identical to the smoke's, and honestly so: the benchmark scores one proposer
genome per checkpoint against the coarse `{nav, battle, discov}` flags, not the continuous
turns-survived metric that showed the spread. At 11 memorized examples the trained proposer's
output doesn't visibly move that coarse score on this single lv6 state. The gradient exists in the
harvest data (proof the two-state/BEATS setup produces real spread to learn from); it has not yet
shown up as a training signal in the benchmark, most likely because (a) the reward the benchmark
reads is binary-flag-based rather than the turns/HP-based metric that actually varies, and (b) 11
rows is too few for the LoRA to learn anything beyond memorizing its training genomes rather than a
genome-conditioned policy. Artifacts: `docs/img/forest_benchmark_trends.png`,
`docs/img/lora_weight_trends.png`.
