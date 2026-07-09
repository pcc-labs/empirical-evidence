# Game-agnostic retrain (sft_v4)

After the [GameProfile refactor](https://github.com/pcc-labs/pokemon-kafka/pull/41) made the agent
playable on Red, Blue, and Yellow, we retrained the local SmolLM3-3B adapter on a **game-agnostic
corpus** — legacy Red telemetry plus fresh Blue and Yellow runs, with every example's system prompt
naming its actual game. This is the successor to the Red-only `sft_v3` / forest-lora weights.

The headline: a base SmolLM3 scores **0% on the Yellow-labelled** ground-truth prompts; the retrained
model scores **75%**, while still improving on held-out Red.

## Corpus

`data/sft_v4` — 535 chat-format examples (seed 42), 484 train / 51 valid (stratified by domain):

| domain | examples | of which Blue/Yellow |
|---|---|---|
| battle-action | 221 | 22 |
| narrator | 214 | 15 |
| move-choice | 56 | 11 |
| battle-outcome | 27 | 4 |
| genome | 17 | 0 |

Sources: ~36k legacy Red events (recorded before the `game` field existed → labelled "Red"), an
800-turn live Blue run (`game=red_blue`), and Yellow runs (`game=yellow`) including the one that
received Pikachu from Oak and won the rival battle — all read at Yellow's shifted RAM addresses.
Battle/move/narrator prompts label themselves per event via `convert_telemetry.game_label`; legacy
events with no field fall back to "Red", keeping the old prompts byte-identical.

## The MLX learning-rate bug (why the first retrain diverged)

The first MLX run **diverged**: validation loss climbed 3.585 → 6.031, train loss spiked past 14.

Root cause: `mlx_lm.lora` applies the config's `learning_rate` as a **constant** when no schedule is
given, and the `MacProfile` used `2e-4` — the *peak* of the CUDA path's cosine schedule. The CUDA/TRL
path (which produced sft_v3) has warmup + cosine decay built in, so the same 2e-4 converged there; the
MLX path applied it flat and diverged.

Fix ([`train_sft.py`](../autotune/train_sft.py), [`config.py`](../autotune/config.py)): the MLX config
now always emits a warmup → cosine-decay schedule (`MacProfile.warmup_steps`, `lr_min`). With warmup 60
→ peak 1e-4 → cosine decay to 1e-6 over 900 iters, training converges cleanly.

## Training convergence

Rank-16 LoRA, 16 layers, 13.4M trainable params (0.437% of 3.08B), MLX backend, `m-large` profile:

| iter | 1 (base) | 150 | 300 | 450 | 600 | 750 | 900 |
|---|---|---|---|---|---|---|---|
| val loss | 3.585 | 0.325 | 0.159 | 0.114 | 0.125 | 0.078 | **0.077** |

Final train loss 0.070 — matching sft_v3's quality, now on the game-agnostic corpus.

## Benchmarks

Greedy (temperature 0) accuracy on the two ground-truth "gated" domains — `battle-outcome`
(win-field) and `move-choice` (move/bucket-field) — scored against game RAM, base SmolLM3 vs the
retrained model. Reproduce with `autotune.eval_heldout --backend mlx`.

**Held-out valid set (truly out-of-sample, Red):**

| domain | base | tuned |
|---|---|---|
| battle-outcome | 0.50 | **1.00** |
| move-choice | 0.40 | **0.60** |

**Game-labelled Blue/Yellow examples** (shows the model conditioning on the named game):

| slice | n | base | tuned |
|---|---|---|---|
| battle-outcome | 4 | 0.75 | **1.00** |
| move-choice | 11 | 0.27 | **0.82** |
| **by game — Yellow** | 4 | **0.00** | **0.75** |
| by game — Red/Blue | 11 | 0.55 | **0.91** |

The base model cannot answer the Yellow-labelled prompts at all (0/4); the retrained model gets 3/4.

**Caveat on sample size + in-sample-ness.** The held-out valid split is stratified by *domain*, not
game, so the few Blue/Yellow gated examples land in *train* — the Blue/Yellow table above is therefore
partly in-sample and small (n = 4 Yellow, 11 Red/Blue). It demonstrates the model *learned* the
game-labelled format, not held-out generalization on Yellow. The held-out Red gate is the
out-of-sample number. Deepening the non-Red signal needs more/longer Blue/Yellow runs (52 of 535
examples are currently non-Red).

## Reproduce

```bash
uv sync --extra mac                      # MLX toolchain (Apple Silicon)

# 1. Build the game-agnostic corpus (Red + Blue + Yellow telemetry)
uv run python -m autotune.convert_telemetry \
  --pk-data ../pokemon-kafka/data \
  --ee-data <dir of fresh Blue/Yellow game telemetry> \
  --rollouts out/brock \
  --out data/sft_v4

# 2. Train (warmup + cosine schedule is now the MLX default)
AUTOTUNE_MAC_PROFILE=m-large \
  uv run python -m autotune.train_sft --data-dir data/sft_v4 --iters 900

# 3. Fuse for serving (mlx_lm.server --adapter-path is broken; serve the fused model)
uv run python -m mlx_lm fuse \
  --model EricFillion/smollm3-3b-mlx --adapter-path out/sft --save-path out/fused

# 4. Benchmark base vs tuned (add --by-game for the per-game split)
uv run python -m autotune.eval_heldout \
  --backend mlx --valid data/sft_v4/valid.jsonl --model out/fused --by-game
```

Artifacts (`out/` and `data/` are gitignored — publish to HF or serve locally):
adapter `out/sft/adapters.safetensors` (~54 MB), fused model `out/fused/` (2.2 GB).
