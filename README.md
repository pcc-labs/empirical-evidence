# empirical-evidence

> *Empirical evidence is information acquired through observation, experimentation, or direct
> experience.* That is what this loop runs on: each generation runs the agent, observes its own
> telemetry, and learns from what actually happened.

A local **Try → Check → Reward → Nudge** training loop that runs on either an NVIDIA GPU
(CUDA + TRL/PEFT, the default on this box) or Apple Silicon via
[MLX](https://github.com/ml-explore/mlx-lm), and **enforces a story** in the
[pokemon-kafka](../pokemon-kafka) agent.

> [!NOTE]
> The loop drives a companion agent, **[`pokemon-kafka`](https://github.com/pcc-labs/pokemon-kafka)**
> (also public). Check both out as siblings and supply a legally-obtained Pokémon Red ROM (never
> included here) to run the loop end-to-end. The `../pokemon-kafka` links throughout this README
> assume that sibling layout.

```
        ┌──────────── empirical-evidence loop ───────────┐
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

empirical-evidence does **not** invent a new narrative. The story is pokemon-kafka's canonical Route-1
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

## Telemetry → SFT corpus → hostable weights (2026-07-05)

Beyond the genome loop, `autotune/convert_telemetry.py` turns raw `pokemon.game.v1` telemetry
(~38k events) into a deterministic multi-domain SFT corpus — battle-outcome, move-choice,
battle-action, genome, narrator — at `data/sft_v3` (590 examples, seed 42, byte-reproducible).
Retraining the SmolLM3-3B LoRA on it and gating with `autotune/eval_heldout.py` (tuned must beat
base on the ground-truth domains) produced the first weights that show real learning rather than
format memorization: tuned **1.00 / 0.56** vs base **0.00 / 0.00** on held-out win-prediction /
type-effectiveness rows. The fused checkpoint + cards live under `out/package/fused/`
(publish runbook: PR [#12](https://github.com/pcc-labs/empirical-evidence/pull/12)).

```bash
uv run python -m autotune.convert_telemetry \
  --pk-data ../pokemon-kafka/data --ee-data data/telemetry --out data/sft_v3 --seed 42
uv run python -m autotune.train_sft --data-dir data/sft_v3
uv run python -m autotune.eval_heldout        # exit 0 = gate passed
uv run python -m autotune.package --skip-eval --seed 42
```

## Integration with pokemon-kafka

The loop runs pokemon-kafka as its environment and feeds what it learns back into how the agent
plays, at three levels (apply the best genome, persist nudges into gameplay, evolve with the
local model). See [docs/pokemon-kafka-integration.md](docs/pokemon-kafka-integration.md) for the
seams, the data contract, and the end-to-end workflow.

For results from the constrained learning experiments (why tuning the 12-parameter genome
saturates on early-game tasks — and the 2026-07-05 addendum on where the signal actually was),
see [docs/experiment-findings.md](docs/experiment-findings.md).

The forest-crossing genome proposer this loop trained is published as
[bdougie/smollm3-forest-lora](https://huggingface.co/bdougie/smollm3-forest-lora); to serve it
locally as an OpenAI-compatible API (for [pi](https://github.com/badlogic/pi-mono) or curl), run
`scripts/serve_forest_lora.sh` — see [docs/serving-forest-lora.md](docs/serving-forest-lora.md).

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
