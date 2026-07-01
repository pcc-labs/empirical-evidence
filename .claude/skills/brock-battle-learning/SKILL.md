---
name: brock-battle-learning
description: Use when running the Pewter-prep speedrun and learning the fastest way to beat Brock (gym 1). Drives the pokemon-kafka agent to Pewter under periodic observation + in-run healing/leveling, captures a pre-Brock state, then runs the empirical-evidence brock loop to learn the fewest-turns win. Requires ROM and uv. Sibling to route1-speedrun-demo.
---

# Brock Battle Learning

## Overview

Where `route1-speedrun-demo` grinds Route 1 for a clean telemetry demo, this skill chases an
**outcome**: beat **Brock** (Pewter Gym, gym 1) in the **fewest turns**, and *learn* the
configuration that does it.

The framing that makes this work (and that the route1 reward could not — see
`empirical-evidence/docs/experiment-findings.md`):

- **The learning target is the battle, not the navigation.** Navigation to Pewter is *plumbing*;
  the thing we optimize is the Brock fight (turns-to-win), which is non-saturated because the
  agent hardcodes **Charmander** (Fire, resisted 0.5× by Rock) against Brock's tanky Onix.
- **The journey is preparation.** You cannot beeline to Pewter in ~5000 turns and win — the agent
  must **level up, pick up/use items (potions), and survive** along the way. Getting to Brock
  *equipped* is part of the learning. Hypothesis: **navigation will have to adjust (grind, heal,
  detour) so the party arrives strong enough to win.**
- **Observe every ~1000 turns.** A long run self-heals in two ways we already have precedent for:
  in-run (battle healing via `hp_heal_threshold`, leveling from wins) and across segments (the
  observer distills telemetry into `pokedex/memory/observations.md`, which the next segment reads
  at startup under `--strategy medium`).

```
Phase A: PREPARE  (pokemon-kafka agent)          Phase B: LEARN  (empirical-evidence)
 drive to Pewter, segmented ~1000-turn run   ->   from the captured pre-Brock state,
 heal + level + collect items on the way          search {lead level × battle genome}
 observe between segments (durable memory)         for the fewest-turns Brock win
 capture pre-Brock save state                      -> out/best_brock.json
```

## Prerequisites

| Check | How |
|-------|-----|
| ROM | `ls "$ROM_PATH"` — Pokemon Red `.gb` |
| uv | `uv --version` |
| empirical-evidence `.env` | `POKEMON_KAFKA_DIR` + `ROM_PATH` set (`cp .env.example .env`) |
| tests green | `cd empirical-evidence && uv run pytest -q` ; `cd ../pokemon-kafka && uv run pytest -q` |
| (optional) Confluent | only if you want live streaming — see `confluent-cloud-setup` |

Run empirical-evidence commands from the `empirical-evidence/` dir, pokemon-kafka commands from `pokemon-kafka/`.

## Phase A — prepare: observed speedrun to Pewter

The goal of this phase is a **pre-Brock save state** with a party strong enough to make the
battle winnable, plus a **path study** of how it got there.

### A1. Clean state
```bash
cd pokemon-kafka
rm -f "rom/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb.ram"   # stale save -> intro fails
```

### A2. Run the journey under observation + self-healing

Use `--strategy medium` so the agent loads `notes.md` / `observations.md` and self-heals, and
`--save-state-on-trainer "brock:..."` so it dumps a state the instant Brock's fight begins.
Run it in **~1000-turn segments**, distilling observations between segments so each segment
starts smarter (the precedent: `SKILL.md` "run 1000 turns" + observational memory).

```bash
ROM="rom/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
STATE="../empirical-evidence/states/pre_brock.state"      # captured the instant Brock's fight begins
CKPT="../empirical-evidence/states/journey.state"          # rolling checkpoint for segment resume
mkdir -p ../empirical-evidence/states

# Segmented run: each segment resumes the journey from the last checkpoint, runs ~1000 turns,
# and distills observations. The journey state carries across segments; the memory grows.
for seg in $(seq 1 15); do
  RESUME=""; [ -f "$CKPT" ] && RESUME="--load-state $CKPT"
  uv run python scripts/agent.py "$ROM" \
    --strategy medium \
    --max-turns 1000 \
    $RESUME \
    --save-state-every "500:$CKPT" \
    --save-state-on-trainer "brock:$STATE" \
    --telemetry-dir data/telemetry \
    --output-json data/telemetry/fitness.json
  uv run python scripts/observe_cli.py        # distill this segment into observations.md
  # Stop early once the pre-Brock state exists (Brock fight reached).
  [ -f "$STATE" ] && { echo "Reached Brock — captured $STATE"; break; }
done
```

Notes:
- `--save-state-every "500:$CKPT"` overwrites a checkpoint twice per segment; the next segment
  resumes from it via `--load-state`, so the journey accumulates instead of restarting from the
  intro. (Delete `$CKPT` to start a fresh journey.)
- `--strategy medium` is what turns on the self-healing memory loop; `low` (empirical-evidence's rollout
  default) does not read notes.
- In-run healing/leveling needs encounters and potions. If the agent reaches Pewter under-leveled,
  that is a real finding — push the level lever in Phase B, or bias the route toward grass to
  grind (see route1-speedrun-demo's grass-loop `routes.json` patch for precedent).
- `BROCK_MAP_ID` can be set once discovered (printed below) to pin Brock detection precisely.

### A3. Study the path (from empirical-evidence)

```bash
cd ../empirical-evidence
# Build the path study from the accumulated telemetry (no re-run). Pass the telemetry dir
# (the one containing game/), not the game/ subdir itself:
uv run python -m autotune.speedrun --from-telemetry ../pokemon-kafka/data/telemetry
cat out/path_study.json   # ordered maps, turns-per-segment, total turns, Brock summary, gym_map_id
```

`path_study.json` is the "study the path" deliverable: where time went, what level/party reached
Brock, and the gym map id. **Even if Pewter isn't reached, this shows where it got stuck** — that
is the navigation signal for the next attempt.

## Phase B — learn: the fewest-turns Brock win

### B1. (optional) generate the level ladder

The matchup lever ("best Pokemon and level") needs pre-Brock states at different levels. Read the
capture nuance in `empirical-evidence/docs/brock-loop.md` first — a party RAM poke only propagates from an
*overworld* state, so for the level lever capture an overworld pre-gym state (`--save-state-on-map
"<gym_map_id>:..."`) rather than the battle-start `pre_brock.state`.

```bash
uv run python -m autotune.party "$ROM_PATH" \
  --in-state states/pre_brock.state --levels 10,12,14,16 --out-dir states/brock
```

### B2. Run the brock loop

```bash
uv run python -m autotune.loop --mode brock --generations 4
```

This searches **{lead level × battle genome}** from the captured state(s), k=1 per config
(deterministic), self-healing the nudge (lost → escalate level; slow win → tune battle params).
Outputs:
- `out/brock/leaderboard.json` — every config → `{reward, won, turns}`, sorted.
- `out/best_brock.json` — the winning `{genome, level}` and its turns.

### B3. Replay / apply the winner

```bash
scripts/apply_brock.sh
```

## What to watch

| Signal | Where | Meaning |
|--------|-------|---------|
| `Saved trainer-battle state ... brock` | Phase A log | reached Brock; pre-Brock state captured |
| `brock_won` / `brock_turns` in fitness | telemetry / path_study | the battle outcome — the thing we optimize |
| `[brock] gen N: ... won=True turns=K` | Phase B log | the loop is finding faster wins |
| reward bands `[0,0.9) < [1,3) < [11,12)` | leaderboard | not-reached < lost < won (see brock-loop.md) |
| stuck on map 12 / wandering backward | Phase A log | navigation, not battle — the known wall |

## The hypothesis this skill tests

> Navigation on the way to Pewter will need to **adjust** for the agent to be **equipped** for
> battle — leveling and acquiring items/Pokemon is part of the learning.

Concretely: if Phase B can't find a Brock win at the level the agent organically reached, the
fix is upstream — grind more (more turns / grass-biased route), heal more (`hp_heal_threshold`),
or arrive with a better matchup. The brock loop's `out/best_brock.json` tells you the **minimum
level** that wins; feed that target back into how Phase A prepares.

## Known wall

`experiment-findings.md` (and a 400-turn smoke) show the baked agent reliably reaches **Route 1**
but struggles across Viridian Forest to Pewter. So Phase A may need many segments, route tuning,
or healing before it ever reaches Brock. The loop degrades gracefully: `path_study.json` records
how far it got, and the reward gives directional credit below Pewter.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Intro failed: still at map=0` | delete the `.gb.ram` save file |
| Never reaches Pewter | more segments; bias route to grind levels; raise `hp_heal_threshold`; check observations.md is growing |
| `pre_brock.state` never written | agent didn't reach a gym-leader-level trainer — it's stuck before Brock (a nav problem) |
| Brock loop: no win at any level | under-prepared party; widen the level ladder, or the matchup needs a super-effective lead (see brock-loop.md Tier-2) |
| Party poke didn't change the fight | state was captured mid-battle — capture an overworld pre-gym state for the level lever |

## stereOS (parallel runs)

pokemon-kafka is set up to run inside a stereOS VM via Master Blaster (`mb up`, `mb attach`; see
`pokemon-kafka/jcard.toml`). To explore many preparation routes / level ladders in parallel,
fan out segmented Phase-A runs across VMs, then run one Phase-B brock loop per captured state.

## Files this skill drives

- pokemon-kafka: `scripts/agent.py` (`--strategy medium`, `--save-state-on-trainer brock:…`),
  `scripts/observe_cli.py`, `pokedex/memory/observations.md`, `notes.md`.
- empirical-evidence: `autotune/speedrun.py`, `autotune/party.py`, `autotune/loop.py --mode brock`,
  `scripts/apply_brock.sh`, `docs/brock-loop.md`.
