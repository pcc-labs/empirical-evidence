# Forest follower → SFT bridge (Option 3)

**Goal:** reach Pewter City, capturing the LoRA's SFT weights on the way. Concretely: cross
Viridian Forest with the landmark-follower, and harvest a forest-reward-keyed SFT buffer from a
genome sweep so the local model has a real gradient to learn instead of the current flat 1/8.

## Background — why the forest reward is flat at 1/8

`forest_story.score_forest` rewards the **in-order** sub-beat frontier (consecutive 1,2,3…).
Beat 2 ("Pick up the Poké Ball") fires on `max_bag_count >= 1`, which reads `bag_count` on
`overworld` events — a field **pokemon-kafka never emits** (`grep bag_count` in pk = nothing). So
beat 2 is a permanent gap, the frontier caps at **1** for every in-forest run regardless of
catchers beaten / sign read / exit reached, and `nudge_sft.build_dataset` returns `[]` (no
gradient → no SFT examples → the LoRA can't learn).

## Why Option 3 (not 1 or 2)

The decisive constraint from prior findings (`memory/weights-for-battle-not-nav.md`): tuning the
**nav genome does not move navigation** — the nav proposer tied the heuristic. So:

- **Option 1** (param-evolution pairs) would teach the genome to navigate — the lever that ties.
  Green pipeline, no learning.
- **Option 2** (behavior-clone the follower's actions) teaches the right thing (nav behavior) but
  has no consumer: the loop's inference seam emits *genomes* (`propose_next_genome`), not
  per-step buttons, and adding an action seam is a pokemon-kafka change (out of scope).
- **Option 3** (chosen): the **follower drives nav** (the broken part the genome can't fix); the
  **genome varies survival** (`hp_run_threshold`, `hp_heal_threshold`, move scoring) — which is
  *battle/survival*, exactly where weights help (`memory/engage-catchers-for-discovery.md`,
  `memory/party-poke-exp-bug.md`). The genome thus sits on a lever that actually moves the forest
  reward, and it plugs straight into the existing `nudge_sft → train_sft → loop` chain.

## Design — four independently-testable units

### 1. De-flatten `forest_story` reward (pure logic, TDD)

- `reward = count of sub-beats reached` (sum of `per_beat`), **not** the strict in-order frontier.
- `per_beat[i] = 1` iff beat `i+1` was reached (no longer "≤ frontier").
- `beats_passed = reward` (the count).
- **Keep** `furthest_beat` / `furthest_beat_name` as the in-order frontier, for human reporting.
- Item beats (2, 5, 7) stay in the 8-beat structure (forward-compatible for when pk emits
  `bag_count`) but no longer gate the gradient.

Worked example (production, no `bag_count`): enter + 2 catchers + sign + exit →
reached `{1,3,4,6,8}` → `reward = 5/8`, `furthest_beat = 1`, `per_beat = (1,0,1,1,0,1,0,1)`.
Faint after one catcher → `reward = 2/8`. A clean climbable gradient.

### 2. Promote the follower to `autotune/forest_follow.py` (IO/subprocess, smoke-tested)

Mirrors `rollout.run_one`'s per-rollout isolation so a batch can sweep genomes in parallel.

- Reads `EVOLVE_PARAMS` (the genome) from env and runs one rollout from a leveled
  forest-entrance state (default `states/forest_lv/lead_lv13.state`).
- **Nav stays hand-driven** (headings + wall-follow + beat advancement) — ported from
  `scratch_forest_follow.py`.
- **Battle delegates to the genome** (the change that creates the gradient):
  - trainers (catchers): **always fought** (`engage-catchers-for-discovery`); genome affects
    move choice + healing.
  - wild encounters: **flee-or-fight via `hp_run_threshold`** — the survival spread lever.
  - healing via `hp_heal_threshold` (note: inert when the bag is empty; see risks).
- **Emits a minimal events list in the `game_events` shape `score_forest` already reads**
  (`overworld` per step with `map_id`; `battle_outcome {battle_type, won}` per fight; `discovery`
  with sign text; `map_change` on exit) — so the same scorer + `load_game_events` work unchanged,
  with **no coupling to pokemon-kafka's collector internals**.
- Returns/persists `{events, verdict, fitness, genome}` per rollout, like `Rollout`.

### 3. Forest-keyed SFT dataset builder in `nudge_sft.py` (pure logic, TDD)

- `ForestWinner(params, forest_verdict, fitness)`.
- Rank by `(reward, trainer_wins, -turns)` — mirrors how `_rank` falls back to a secondary signal
  when the primary ties (`trainer_wins` from `verdict.signals`, `turns` from `fitness`).
- `build_forest_dataset(winners)` / `assemble_forest_corpus(winners_by_state)`: within one start
  state, rejection-sample weaker genome → strongest genome (`weak → best`); return `[]` when
  there is no gradient (fewer than two crossings, or all tied at the top). Same shape as
  `build_dataset`.
- User-turn prompt: a forest-shaped mutation prompt describing the sub-beats reached + the genome,
  so train and (future) inference shapes match. Assistant target = the flat best-genome JSON
  (parseable by `nudge_steer.parse_genome_response`). Write via the existing `write_sft_data`.

### 4. End-to-end + scope

- Wire a small `forest_follow` genome sweep → score with the de-flattened `forest_story` →
  `assemble_forest_corpus` → `write_sft_data`. Confirm: reward **spreads** across genomes and the
  buffer is **non-empty**. That mini-run is "reach Pewter + capture weights" in miniature.
- **In scope:** units 1–3 + the end-to-end harvest producing `train.jsonl`/`valid.jsonl`.
- **Out of scope:** wiring the trained LoRA *back* into `forest_loop` inference (it currently
  mutates randomly via `mutate_nav_params`); any pokemon-kafka change; rich winprob rows from the
  follower (bonus, later).

## Implementation risks — verify before building on them

1. **Does `PokemonAgent` honor `EVOLVE_PARAMS` for battle when constructed in-process by the
   follower?** pk's `evolve.py` sets it for the subprocess agent; confirm the battle path
   (move scoring + heal threshold) actually reads it. Spike before building unit 2's battle seam.
2. **Does survival genuinely vary across genomes on the follower route?** If the bag is empty,
   `hp_heal_threshold` is inert and only `hp_run_threshold` moves the needle. If every genome
   survives the careful route, rewards tie and we're flat again. Mitigation: harsher route / a
   level low enough that `hp_run_threshold` decisions change the outcome. Verify with a 2–3 genome
   spike before scaling the sweep.

## Test plan

- Unit 1: new tests locking count-of-reached where it diverges from the frontier (enter+sign →
  reward 2; enter+2 catchers+exit, no bag → reward 4); existing tests keep passing (frontier moves
  to `furthest_beat`).
- Unit 3: gradient present → pairs built weak→best; no gradient / single crossing → `[]`; pairs
  are within-state only; assistant target is flat parseable genome JSON.
- Unit 2: smoke only (needs ROM) — a 2–3 genome spike asserting reward spread + non-empty buffer.
