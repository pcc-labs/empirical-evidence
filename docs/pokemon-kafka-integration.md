# autotune ↔ pokemon-kafka integration

This document explains how autotune and [pokemon-kafka](../../pokemon-kafka) work together: how
the training loop runs the agent, scores it against a story, and feeds what it learns back into
how the agent plays.

## Roles

autotune is the **training loop**. pokemon-kafka is the **environment**. autotune runs the
pokemon-kafka agent, verifies each run against an ordered story, and reinforces what stayed
on-story. It does not fork or vendor pokemon-kafka; it talks to it through stable seams.

```
        ┌──────────────── autotune loop ────────────────┐
 story  │  Try            Check + Reward         Nudge   │
 spec ─►│  rollout.py ──► verifier.py ──► ┌ nudge_sft.py │──► new genome /
        │  run pk agent   per-beat        └ nudge_steer  │    adapter
        │  (N tries)      pass=1/fail=0                   │
        └────────────────── loop back to Try ────────────┘
```

## The core loop, mapped to pokemon-kafka

| Stage | Meaning | Where it happens |
|-------|---------|------------------|
| **Try** | The agent acts (game inputs / tool calls). | `rollout.py` runs `pokemon-kafka/scripts/agent.py` N times. |
| **Check** | A verifier scores the run against the story. | `verifier.py` reads the run's telemetry. |
| **Reward** | Per-beat `pass=1 / fail=0`, in order. | `verifier.py` produces a `RolloutVerdict`. |
| **Nudge** | Reinforce what passed. | `nudge_sft.py` (train) and/or `nudge_steer.py` (steer). |

The story itself is pokemon-kafka's own: chapter ordering from `scripts/evolve.py::MAP_PROGRESS`
and per-beat waypoints from `references/routes.json`. autotune mirrors the ordering in
`story.py` and reads the waypoints at runtime.

## The seams (data contract)

autotune and pokemon-kafka exchange data through four stable interfaces. None of them require
importing the other project at runtime.

| Seam | Direction | Shape |
|------|-----------|-------|
| `EVOLVE_PARAMS` env var | autotune → pk | JSON genome the agent applies at startup. |
| Telemetry JSONL + fitness | pk → autotune | `<telemetry-dir>/game/<date>.jsonl` events + `--output-json` fitness. |
| Genome block in `notes.md` | autotune → pk | A marked JSON block the agent reads at startup (L2). |
| `out/best_genome.json` | autotune → you | The best genome found, for applying to a real run (L1). |

### The genome block format

autotune writes this block into `pokemon-kafka/notes.md`. pokemon-kafka reads the last such
block at agent startup.

```
<!-- autotune:genome
{"stuck_threshold": 8, "door_cooldown": 10, ...}
-->
```

## Three ways learnings flow back

The loop runs pokemon-kafka regardless. These three levels control how what it learns persists
into normal pokemon-kafka runs.

### L1: apply the winning genome

The loop tracks the best genome across generations and writes `out/best_genome.json`.

```bash
# Train, then apply the best genome to a long real run:
uv run python -m autotune.loop --nudge both
scripts/apply_genome.sh 4000
```

`apply_genome.sh` reads `out/best_genome.json` and runs the pokemon-kafka agent with that genome
via `EVOLVE_PARAMS`. No pokemon-kafka changes are involved.

### L2: nudges into gameplay

Each steer nudge writes the genome block into `pokemon-kafka/notes.md`. The pokemon-kafka agent
reads that block at startup and uses it as its parameter baseline. The `EVOLVE_PARAMS` env var
still overrides it, so behavior is unchanged when no block is present.

```bash
# autotune writes pk/notes.md during the loop:
uv run python -m autotune.loop --nudge steer

# pokemon-kafka picks it up with zero flags:
cd ../pokemon-kafka && uv run scripts/agent.py rom/*.gb
```

Point the block somewhere else with `AUTOTUNE_NOTES_PATH` (for example, to keep it inside
autotune's `out/` during experiments).

The read side lives in pokemon-kafka: `scripts/autotune_bridge.py::load_genome_from_notes`,
called from `scripts/agent.py`.

### L3: pokemon-kafka evolves with the local model

pokemon-kafka's own `scripts/evolve.py` can use autotune's locally-trained model as its
mutation proposer instead of Claude.

```bash
cd ../pokemon-kafka
uv run scripts/evolve.py rom/*.gb --llm local
```

`--llm local` calls `autotune_bridge.make_local_llm_fn`, which drives autotune's model through
`mlx_lm generate`. No API key is needed. The default `--llm anthropic` and `--no-llm` behaviors
are unchanged.

## End-to-end workflow

```bash
cd autotune
cp .env.example .env          # set POKEMON_KAFKA_DIR and ROM_PATH

# 1. Run the loop. Writes out/best_genome.json and pk/notes.md.
uv run python -m autotune.loop --nudge both --generations 5 --n 4

# 2a. Apply the best genome to a real run (L1).
scripts/apply_genome.sh 4000

# 2b. Or let pokemon-kafka pick up the genome on its own (L2).
cd ../pokemon-kafka && uv run scripts/agent.py rom/*.gb

# 2c. Or let pokemon-kafka evolve with the local model (L3).
uv run scripts/evolve.py rom/*.gb --llm local
```

## Files involved

### In autotune
| File | Role |
|------|------|
| `rollout.py` | Run the pk agent N times (Try). |
| `verifier.py`, `story.py` | Score runs against the story (Check + Reward). |
| `nudge_steer.py` | Write `notes.md` genome block + propose the next genome (L2 write side). |
| `nudge_sft.py`, `train_sft.py` | Train a local LoRA adapter on what passed. |
| `loop.py` | Orchestrate, track the best genome, write `out/best_genome.json` (L1). |
| `scripts/apply_genome.sh` | Run pokemon-kafka with the best genome (L1). |

### In pokemon-kafka
| File | Role |
|------|------|
| `scripts/autotune_bridge.py` | Read the notes genome (L2); build the local proposer (L3). |
| `scripts/agent.py` | Seed `evolve_params` from the notes genome at startup (L2). |
| `scripts/evolve.py` | `--llm local` uses autotune's model as the proposer (L3). |

## Design modes: frozen (current) vs live (future)

**Frozen (current, default).** autotune tunes offline. It runs the agent many times, scores each
run, and writes a genome into `notes.md` and `out/best_genome.json`. The speedrun then loads that
genome once at startup and plays with no autotune code running. The genome crosses between phases
as data, not as running code. This is the only mode that exists today.

**Live (future, not built).** autotune would steer the agent during the speedrun, updating its
behavior as it plays rather than only at startup. This is a different, online design. The frozen
path stays the default; live steering would be opt-in. Tracked in
[issue #2](https://github.com/pcc-labs/autotune/issues/2).

## Status

SFT-first MVP. The reward is binary per-beat `pass=1 / fail=0`, consumed by rejection-sampling
SFT. GRPO (true policy-gradient RL), an LLM-judge eval rig, and serving are deferred behind the
same Nudge interface.
