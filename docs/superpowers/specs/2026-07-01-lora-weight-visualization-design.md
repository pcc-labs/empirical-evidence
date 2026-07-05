# LoRA weight visualization

**Goal:** visualize how the LoRA adapter weights in `out/sft/` change across a training run, so
it's possible to see which layers `nudge_sft.py`/`train_sft.py` actually moved and whether
training is still progressing or has plateaued by the final checkpoint.

## Background

`out/sft/` already contains real checkpoints from an MLX LoRA run (`0000100_adapters.safetensors`,
`0000200_adapters.safetensors`, `0000300_adapters.safetensors`, and the final
`adapters.safetensors`), rank 16, 16 layers. Each checkpoint stores per-layer `lora_a`/`lora_b`
pairs (MLX naming) — confirmed via `safetensors.safe_open`, e.g.
`model.layers.20.mlp.down_proj.lora_a (11008, 16)` / `...lora_b (16, 2048)`. The effective weight
delta for a given LoRA matrix is `B @ A`; its Frobenius norm is a scalar proxy for "how much this
matrix changed from the base model." There is currently no tooling in the repo to inspect this —
this is a new, standalone read-only analysis tool, not a change to the loop.

The repo supports two backends (`cuda` via HF/PEFT, `mlx` via `mlx-lm`), whose adapter files use
different key naming (MLX: `lora_a`/`lora_b`; PEFT: `lora_A.weight`/`lora_B.weight`). The tool
reads raw `.safetensors` files directly (via the `safetensors` library, not `torch`/`mlx-lm`), so
it has no backend-specific runtime dependency and should handle either naming convention.

## Design — three independently-testable units

### 1. `autotune/weights_viz.py` — pure norm computation (TDD)

- `load_lora_deltas(path: Path) -> dict[str, float]` — opens one `.safetensors` checkpoint, pairs
  up each `lora_a`/`lora_b` (or `lora_A.weight`/`lora_B.weight`) by matching normalized key
  prefixes, computes `norm(B @ A)` (Frobenius) for each pair, keyed by a normalized name like
  `layers.20.mlp.down_proj`. Raises a clear error naming the offending key if a matrix is missing
  its pair (no silent skip).
- `aggregate_by_layer(deltas: dict[str, float]) -> dict[int, float]` — sums norms across all
  module types (`mlp.down_proj`, `self_attn.q_proj`, etc.) within each layer index, producing one
  aggregate norm per layer (16 numbers instead of 112).

Both functions are pure (numpy in, numpy/dict out) and take no torch/mlx dependency — testable
with tiny synthetic safetensors files written to `tmp_path`.

### 2. `plot_norm_trends` — matplotlib driver

- `plot_norm_trends(steps: list[int], per_checkpoint_layer_norms: list[dict[int, float]], out_path: Path) -> None`
- One line per layer index (16 lines), x-axis = checkpoint step, y-axis = aggregate `B @ A` norm
  for that layer at that step. Answers "which layers are still moving vs. flat by the final step."
- Matplotlib-driven, so — like `train_sft`/`generate`/`package` — this is exercised manually, not
  covered by pytest.

### 3. CLI entry point

- `uv run python -m autotune.weights_viz [--adapter-dir out/sft] [--out out/lora_weight_trends.png]`
- Discovers checkpoints in `--adapter-dir` matching `NNNNNNN_adapters.safetensors`, sorted by the
  numeric step in the filename. The bare `adapters.safetensors` (final save) is a distinct file
  from its same-step numbered checkpoint (verified: different bytes), so it is plotted as its own
  trailing point labeled `"final"` on the x-axis rather than assumed to share a step number.
  Errors clearly if fewer than 2 checkpoints are found (nothing to trend).
- Runs `load_lora_deltas` → `aggregate_by_layer` per checkpoint, assembles the step-indexed series,
  calls `plot_norm_trends`.

## Dependencies

Add `matplotlib>=3.8` to `[project.dependencies]` in `pyproject.toml` (not dev-only — the script
is meant to be run standalone via `uv run`).

## Out of scope

- No changes to `loop.py`, `nudge_sft.py`, `train_sft.py`, or any pokemon-kafka code.
- No automatic invocation from the training loop (this is a manual, post-hoc analysis tool over
  checkpoints already on disk).
- No heatmap-of-raw-weights view, no notebook — line-chart-of-norms only, per the chosen design.

## Test plan

- `load_lora_deltas`: given a synthetic safetensors file with 2 known `lora_a`/`lora_b` pairs
  (MLX naming) and one with PEFT naming, returns the expected per-key norms; a matrix missing its
  pair raises with the offending key in the message.
- `aggregate_by_layer`: given a fixed multi-layer, multi-module `deltas` dict, sums correctly per
  layer.
- CLI/`plot_norm_trends`: smoke-tested manually against `out/sft/` (needs the real checkpoints and
  matplotlib), not covered by pytest.
