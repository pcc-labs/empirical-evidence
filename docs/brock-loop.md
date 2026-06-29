# The self-healing Brock loop

A `Try → Check → Reward → Nudge` loop whose objective is an **outcome**, not a fixed path:
speedrun to Pewter City, then **beat Brock (gym 1) in the fewest turns**. It is the answer to
the dead end documented in [experiment-findings.md](experiment-findings.md), where the reward
saturated flat because the story was a hardcoded `routes.json` path and the 12-param genome had
no leverage over the outcome.

## Why this is learnable where route1 was not

The findings doc identified three traps. This loop is built to avoid each:

| Trap (route1) | Fix (brock) |
|---|---|
| **Saturation** — per-beat reward flat at 5.0 | **Continuous, banded reward** on turns-to-beat-Brock; dense in every band (see below). |
| **Low leverage** — genome didn't change outcomes | **Real levers:** battle move-scoring (`score_move = power × acc × type_eff`) and the matchup (lead level/species). Against Brock's tanky Onix these decide turn count *and* win/loss. |
| **Determinism** — k>1 replays identical | **k=1 over distinct configs.** The gradient is across `(state, genome)` configs, not replicates. |

The agent hardcodes **Charmander** — Fire, resisted 0.5× by Rock — so "what beats Brock fastest"
is genuinely open, not pre-solved.

## The reward (`autotune/brock.py`)

Disjoint bands ⇒ strictly monotone, with a dense gradient inside each band (no flat plateau to
random-walk on):

```
didn't reach Pewter  ->  nav_progress * 0.9            # [0, 0.9)   directional (in-order frontier, NOT maps_visited)
reached but lost     ->  1.0 + damage_frac * 2.0       # [1.0, 3.0) how much of Brock's team it chipped
won                  ->  1.0 + 10.0 + (50 - turns)/50  # [11, 12)   strictly faster = strictly higher
```

`nav_progress` reuses the verifier's directional in-order beat frontier (the findings doc proved
`maps_visited` rewards *backward* wandering). White-out is a diagnostic / secondary sort key,
kept out of the scalar so it can never cross a band boundary.

## Running it

```bash
# 0. Setup
cp .env.example .env          # POKEMON_KAFKA_DIR + ROM_PATH
uv sync

# 1. Phase A — speedrun to Pewter, study the path, capture a pre-Brock state.
uv run python -m autotune.speedrun --max-turns 8000
#   -> out/path_study.json   (ordered maps, turns-per-segment, total turns, Brock summary)
#   -> states/pre_brock.state (dumped the instant Brock's fight begins)

# 2. (optional) Phase B level lever — generate level-variant states.
#    NOTE the capture nuance below before using this.
uv run python -m autotune.party "$ROM_PATH" --in-state states/pre_brock.state \
    --levels 10,12,14,16 --out-dir states/brock

# 3. Phase B/C — search {level × battle genome} for the fewest-turns win.
uv run python -m autotune.loop --mode brock --generations 4
#   -> out/brock/leaderboard.json  (every config -> reward/won/turns, sorted)
#   -> out/best_brock.json

# 4. Replay the winning config.
scripts/apply_brock.sh
```

## The save-state capture nuance (important for the level lever)

`party.py` writes the **party struct** (level + recomputed HP/stats). When a Pokemon is sent into
battle the game copies the party struct into a transient battle block (`0xD0xx`), so a party poke
only propagates into the Brock fight if the state was captured **in the overworld, before the
fight**:

- **`states/pre_brock.state`** is captured at *battle start* (`--save-state-on-trainer brock:...`).
  It is correct for the **battle-genome lever** (optimize move scoring at the organic matchup),
  but a `party.py` poke on it will **not** change the active battler.
- For the **level lever**, capture an *overworld* pre-gym state instead (e.g.
  `--save-state-on-map "<gym_interior_map_id>:..."`, the gym map id is reported in
  `out/path_study.json` as `brock.gym_map_id`), poke that, and let the agent step into Brock.

The loop consumes whatever states exist in `states/brock/` (a `*lv<N>.state` ladder) and falls
back to the single `states/pre_brock.state`. With one state it optimizes the genome; with a
ladder it also climbs levels via the self-healing router (lost → escalate level).

## Known risk: does the baked agent reach Pewter?

[experiment-findings.md](experiment-findings.md) (Experiment 1) observed every rollout reaching
**Route 1 and no further** within its turn budget, and Experiment 3 saw the navigator wander
*backward* from Route 1. Crossing Viridian Forest (a maze) to Pewter is materially harder than
Route 1. So Phase A may not reach Brock on the first try.

The loop degrades gracefully rather than failing:

- `out/path_study.json` records **how far it got and where time went** — that *is* the
  "study the path" deliverable even when Pewter isn't reached.
- The reward gives directional `nav_progress` credit below Pewter, and the self-healing router
  nudges nav params when Pewter isn't reached, so the loop keeps making progress instead of
  flat-lining.

## Results

_To be filled in after the end-to-end smoke run on the ROM:_

- Path study: maps reached, turns-per-segment, total turns to Pewter, gym map id.
- Brock leaderboard: best `{lead Pokemon, level, battle genome} → turns`.
- The fewest-turns answer to "how to beat Brock."
