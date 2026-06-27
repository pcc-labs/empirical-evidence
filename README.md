# autotune

A local **Try → Check → Reward → Nudge** training loop that runs entirely on Apple Silicon via
[MLX](https://github.com/ml-explore/mlx-lm), and **enforces a story** in the
[pokemon-kafka](../pokemon-kafka) agent.

```
        ┌──────────────── autotune loop ────────────────┐
 story  │  Try            Check + Reward         Nudge   │
 spec ─►│  rollout.py ──► verifier.py ──► ┌ nudge_sft.py │──► new genome /
        │  run pk agent   per-beat        └ nudge_steer  │    adapter
        │  (N tries)      pass=1/fail=0                   │
        └────────────────── loop back to Try ────────────┘
```

It is modeled on **overdub** (a tapes→SFT/DPO fine-tuning pipeline) but swaps CUDA/cloud-GPU
training for **local LoRA training on an M-series Mac**, and is organized around "the core loop":

> **Try** (agent acts) → **Check** (verifier runs the tests) → **Reward** (pass=1, fail=0) →
> **Nudge** (do more of what passed) → loop back to Try. *That arrow is the entire training process.*

## The story is pokemon-kafka's own

autotune does **not** invent a new narrative. The story is pokemon-kafka's canonical Route-1
progression, assembled from concepts that already exist there:

- chapter ordering = `MAP_PROGRESS` (`pokemon-kafka/scripts/evolve.py`), and
- per-beat waypoints/names = `references/routes.json`.

`verifier.py` turns each rollout's telemetry into a **per-beat pass=1/fail=0** reward against that
ordered story: a story-shaped, verifiable signal, not a scalar fitness blob.

## Two Nudge backends (run either or both)

| `--nudge` | What it does |
|---|---|
| `sft`   | Rejection-sampling LoRA SFT of a local MLX model (`smollm3-3b-mlx`) on the rollouts that passed, then uses that model to propose the next genome. |
| `steer` | Mutates the agent's `EVOLVE_PARAMS` genome from what passed and writes a nudge to `notes.md` (no weight training). |
| `both`  | Both, sharing one loop (default). |

pokemon-kafka is driven through its already-wired `EVOLVE_PARAMS` seam, so **no pokemon-kafka edits
are required.**

## Quickstart

```bash
cp .env.example .env          # set POKEMON_KAFKA_DIR and ROM_PATH
uv sync

# One full generation, both nudge backends:
uv run python -m autotune.loop --generations 1 --n 3 --nudge both
# or: scripts/loop.sh 1 3 both
```

### Pieces individually
```bash
uv run python -m autotune.rollout --n 1 --max-turns 500     # Try (produces telemetry + fitness)
uv run python -m autotune.train_sft --iters 50              # Nudge #1 (LoRA SFT) on existing data/sft
uv run python -m autotune.generate --prompt-beat route1     # ask the trained model for a genome
```

## Integration with pokemon-kafka

The loop runs pokemon-kafka as its environment and feeds what it learns back into how the agent
plays, at three levels (apply the best genome, persist nudges into gameplay, evolve with the
local model). See [docs/pokemon-kafka-integration.md](docs/pokemon-kafka-integration.md) for the
seams, the data contract, and the end-to-end workflow.

## Layout
| File | Role |
|---|---|
| `config.py` | `MacProfile` (MLX knobs), model, story, loop config; env-overridable. |
| `story.py` | Ordered `StoryBeat`s from `MAP_PROGRESS` + `routes.json`. |
| `verifier.py` | Check + Reward: telemetry → per-beat pass/fail + on-story reward. |
| `rollout.py` | Try: run the pk agent N times via `EVOLVE_PARAMS`, collect telemetry. |
| `selection.py` | Rejection-sampling winner selection ("do more of what passed"). |
| `nudge_sft.py` / `train_sft.py` | Build SFT data from winners; train a LoRA adapter via MLX-LM. |
| `nudge_steer.py` | Propose the next genome + write a `notes.md` nudge. |
| `generate.py` | Query the local model to propose a genome (closes path 1). |
| `loop.py` | The orchestrator. |
| `genome.py` | The `EVOLVE_PARAMS` genome (mirrors evolve.py), decoupled from pk's import path. |

## Requirements
- Apple Silicon Mac (MLX). `smollm3-3b-mlx` (~5 GB) is the default and fits comfortably in 16 GB+.
  gpt-oss-120b (~65 GB) does **not** fit a 36 GB machine; swap models via `AUTOTUNE_BASE_MODEL`.
- `uv`, and a legally-obtained Pokemon Red ROM for real rollouts.

## Status
SFT-first MVP. GRPO (true policy-gradient RL), an LLM-judge eval rig, and serving are deferred
behind the same Nudge interface.
