# LoRA weight visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build a standalone, read-only tool that plots per-layer LoRA weight-delta norms across
the training checkpoints in `out/sft/`, so it's visible which layers moved and whether training
had plateaued by the final step.

**Architecture:** a single new module, `autotune/weights_viz.py`, with pure norm-computation
functions (unit-tested), a matplotlib plotting function (manually smoke-tested), and a CLI entry
point (`python -m autotune.weights_viz`). No changes to `loop.py`, `nudge_sft.py`, `train_sft.py`,
or pokemon-kafka.

**Tech Stack:** Python 3.11+, `safetensors` (already a transitive dep via `mlx-lm`/`peft`, read
directly with `framework="numpy"` so no `torch`/`mlx-lm` import is required), `numpy`, new dep
`matplotlib`, `pytest`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-01-lora-weight-visualization-design.md`
- Add `matplotlib>=3.8` to `[project.dependencies]` in `pyproject.toml` (not dev-only).
- Read `.safetensors` files directly via the `safetensors` library — no `torch`/`mlx-lm` runtime
  dependency in this module.
- Must handle both MLX adapter naming (`...lora_a` / `...lora_b`) and PEFT/HF naming
  (`...lora_A.weight` / `...lora_B.weight`).
- Ruff: `E, F, I, W`, line-length 100 (`uv run ruff check`).
- Pure logic (`load_lora_deltas`, `aggregate_by_layer`, `discover_checkpoints`) gets pytest unit
  tests; the matplotlib driver (`plot_norm_trends`) and CLI `main()` are smoke-tested manually,
  matching the convention for `train_sft`/`generate`/`package`.
- No changes to `autotune/loop.py`, `autotune/nudge_sft.py`, `autotune/train_sft.py`, or any
  pokemon-kafka file.

---

### Task 1: Core delta-norm computation (`load_lora_deltas`)

**Files:**
- Create: `autotune/weights_viz.py`
- Test: `tests/test_weights_viz.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `_normalize_key(key: str) -> tuple[str, str] | None` — returns `(prefix, side)` where `side`
    is `"a"` or `"b"`, or `None` if `key` isn't a LoRA tensor.
  - `_delta_norm(a: np.ndarray, b: np.ndarray) -> float` — Frobenius norm of the effective delta
    matrix, shape-detecting whether `a @ b` or `b @ a` is the valid orientation.
  - `load_lora_deltas(path: Path) -> dict[str, float]` — maps normalized prefix (e.g.
    `"layers.0.mlp.down_proj"`) to delta norm, for every LoRA pair in the checkpoint at `path`.
    Raises `ValueError` naming the offending prefix if a matrix is missing its pair.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_weights_viz.py`:

```python
import numpy as np
import pytest
from safetensors.numpy import save_file

from autotune.weights_viz import load_lora_deltas


def test_load_lora_deltas_mlx_naming(tmp_path):
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    expected_norm = float(np.linalg.norm(a @ b))

    path = tmp_path / "checkpoint.safetensors"
    save_file(
        {
            "model.layers.0.mlp.down_proj.lora_a": a,
            "model.layers.0.mlp.down_proj.lora_b": b,
        },
        str(path),
    )

    deltas = load_lora_deltas(path)

    assert deltas == pytest.approx({"layers.0.mlp.down_proj": expected_norm})


def test_load_lora_deltas_peft_naming(tmp_path):
    lora_a = np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    lora_b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    expected_norm = float(np.linalg.norm(lora_b @ lora_a))

    path = tmp_path / "checkpoint.safetensors"
    save_file(
        {
            "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.model.layers.5.self_attn.q_proj.lora_B.weight": lora_b,
        },
        str(path),
    )

    deltas = load_lora_deltas(path)

    assert deltas == pytest.approx({"layers.5.self_attn.q_proj": expected_norm})


def test_load_lora_deltas_missing_pair_raises(tmp_path):
    a = np.zeros((4, 2), dtype=np.float32)
    path = tmp_path / "checkpoint.safetensors"
    save_file({"model.layers.2.mlp.gate_proj.lora_a": a}, str(path))

    with pytest.raises(ValueError, match="layers.2.mlp.gate_proj"):
        load_lora_deltas(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_weights_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autotune.weights_viz'`

- [ ] **Step 3: Write the minimal implementation**

Create `autotune/weights_viz.py`:

```python
"""Compute and plot LoRA weight-delta norms from safetensors checkpoints."""

import re
from pathlib import Path

import numpy as np
from safetensors import safe_open

_MLX_SUFFIX = re.compile(r"\.lora_(a|b)$")
_PEFT_SUFFIX = re.compile(r"\.lora_(A|B)\.weight$")


def _normalize_key(key: str) -> tuple[str, str] | None:
    match = _MLX_SUFFIX.search(key)
    if match is None:
        match = _PEFT_SUFFIX.search(key)
    if match is None:
        return None
    prefix = key[: match.start()]
    prefix = prefix.removeprefix("base_model.model.")
    prefix = prefix.removeprefix("model.")
    return prefix, match.group(1).lower()


def _delta_norm(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[1] == b.shape[0]:
        return float(np.linalg.norm(a @ b))
    if b.shape[1] == a.shape[0]:
        return float(np.linalg.norm(b @ a))
    raise ValueError(f"lora_a shape {a.shape} incompatible with lora_b shape {b.shape}")


def load_lora_deltas(path: Path) -> dict[str, float]:
    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            normalized = _normalize_key(key)
            if normalized is None:
                continue
            prefix, side = normalized
            pairs.setdefault(prefix, {})[side] = f.get_tensor(key)

    deltas: dict[str, float] = {}
    for prefix, sides in pairs.items():
        if "a" not in sides:
            raise ValueError(f"{prefix} is missing its lora_a half in {path}")
        if "b" not in sides:
            raise ValueError(f"{prefix} is missing its lora_b half in {path}")
        deltas[prefix] = _delta_norm(sides["a"], sides["b"])
    return deltas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_weights_viz.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add autotune/weights_viz.py tests/test_weights_viz.py
git commit -m "feat: compute LoRA weight-delta norms from safetensors checkpoints"
```

---

### Task 2: Per-layer aggregation (`aggregate_by_layer`)

**Files:**
- Modify: `autotune/weights_viz.py`
- Test: `tests/test_weights_viz.py`

**Interfaces:**
- Consumes: prefixes produced by `load_lora_deltas` (Task 1), always of the form
  `"layers.<N>.<module>.<proj>"`.
- Produces: `aggregate_by_layer(deltas: dict[str, float]) -> dict[int, float]` — sums norms across
  module types within each layer index; ignores any key not matching `"layers.<N>."`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weights_viz.py`:

```python
from autotune.weights_viz import aggregate_by_layer


def test_aggregate_by_layer_sums_within_layer_and_ignores_unmatched():
    deltas = {
        "layers.0.mlp.down_proj": 1.0,
        "layers.0.self_attn.q_proj": 2.0,
        "layers.1.mlp.up_proj": 5.0,
        "not_a_layer_key": 9.0,
    }

    assert aggregate_by_layer(deltas) == {0: 3.0, 1: 5.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weights_viz.py -k aggregate_by_layer -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate_by_layer'`

- [ ] **Step 3: Write the minimal implementation**

Add to `autotune/weights_viz.py` (after `load_lora_deltas`):

```python
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def aggregate_by_layer(deltas: dict[str, float]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for prefix, norm in deltas.items():
        match = _LAYER_RE.match(prefix)
        if match is None:
            continue
        layer = int(match.group(1))
        totals[layer] = totals.get(layer, 0.0) + norm
    return totals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weights_viz.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add autotune/weights_viz.py tests/test_weights_viz.py
git commit -m "feat: aggregate LoRA weight-delta norms by layer"
```

---

### Task 3: Plotting driver (`plot_norm_trends`) + matplotlib dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `autotune/weights_viz.py`

**Interfaces:**
- Consumes: `list[str]` of checkpoint labels and `list[dict[int, float]]` — one per-layer norm
  dict per checkpoint, in the same order as the labels (shape produced by `aggregate_by_layer`,
  Task 2, applied once per checkpoint).
- Produces: `plot_norm_trends(labels: list[str], per_checkpoint_layer_norms: list[dict[int, float]], out_path: Path) -> None`
  — writes a PNG to `out_path`, one line per layer index across the checkpoint labels.

This task has no pytest coverage per the Global Constraints (matplotlib driver, smoke-tested
manually) — steps are implementation + manual verification instead of TDD.

- [ ] **Step 1: Add the matplotlib dependency**

In `pyproject.toml`, add `"matplotlib>=3.8",` to the `dependencies` list (alongside `torch`,
`transformers`, etc.), then sync:

Run: `uv sync`
Expected: resolves and installs `matplotlib` with no errors.

- [ ] **Step 2: Implement `plot_norm_trends`**

Add to `autotune/weights_viz.py`:

```python
def plot_norm_trends(
    labels: list[str],
    per_checkpoint_layer_norms: list[dict[int, float]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({layer for norms in per_checkpoint_layer_norms for layer in norms})
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in layers:
        ys = [norms.get(layer, 0.0) for norms in per_checkpoint_layer_norms]
        ax.plot(labels, ys, marker="o", label=f"layer {layer}")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("aggregate weight-delta norm per layer")
    ax.set_title("LoRA weight delta norms by layer across checkpoints")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

- [ ] **Step 3: Manually smoke-test the plot**

Run:

```bash
uv run python -c "
from pathlib import Path
from autotune.weights_viz import plot_norm_trends

plot_norm_trends(
    ['100', '200', 'final'],
    [{0: 1.0, 1: 2.0}, {0: 1.5, 1: 1.8}, {0: 2.0, 1: 1.6}],
    Path('/tmp/smoke_trend.png'),
)
"
```

Expected: no error, and `/tmp/smoke_trend.png` exists and is non-empty
(`ls -la /tmp/smoke_trend.png` shows a file size > 0). Open it and confirm it shows 2 lines
(layer 0, layer 1) across 3 x-axis ticks.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock autotune/weights_viz.py
git commit -m "feat: add LoRA weight-delta trend plotting"
```

---

### Task 4: Checkpoint discovery + CLI entry point

**Files:**
- Modify: `autotune/weights_viz.py`
- Test: `tests/test_weights_viz.py`

**Interfaces:**
- Consumes: `load_lora_deltas` (Task 1), `aggregate_by_layer` (Task 2), `plot_norm_trends`
  (Task 3).
- Produces:
  - `discover_checkpoints(adapter_dir: Path) -> list[tuple[str, Path]]` — numbered
    `NNNNNNN_adapters.safetensors` files sorted by step, as `(str(step), path)`, followed by
    `(adapter_dir / "adapters.safetensors", "final")` if it exists.
  - `main(argv: list[str] | None = None) -> None` — CLI entry point for
    `python -m autotune.weights_viz`.

- [ ] **Step 1: Write the failing test for `discover_checkpoints`**

Append to `tests/test_weights_viz.py`:

```python
from autotune.weights_viz import discover_checkpoints


def test_discover_checkpoints_sorts_numbered_and_appends_final(tmp_path):
    (tmp_path / "0000200_adapters.safetensors").write_bytes(b"")
    (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_bytes(b"")

    checkpoints = discover_checkpoints(tmp_path)

    assert [label for label, _ in checkpoints] == ["100", "200", "final"]
    assert checkpoints[0][1] == tmp_path / "0000100_adapters.safetensors"
    assert checkpoints[2][1] == tmp_path / "adapters.safetensors"


def test_discover_checkpoints_no_final_file(tmp_path):
    (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")

    checkpoints = discover_checkpoints(tmp_path)

    assert [label for label, _ in checkpoints] == ["100"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_weights_viz.py -k discover_checkpoints -v`
Expected: FAIL with `ImportError: cannot import name 'discover_checkpoints'`

- [ ] **Step 3: Implement `discover_checkpoints` and `main`**

Add to `autotune/weights_viz.py`:

```python
import argparse

_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def discover_checkpoints(adapter_dir: Path) -> list[tuple[str, Path]]:
    numbered: list[tuple[int, Path]] = []
    for candidate in adapter_dir.glob("*_adapters.safetensors"):
        match = _CHECKPOINT_RE.match(candidate.name)
        if match is not None:
            numbered.append((int(match.group(1)), candidate))
    numbered.sort(key=lambda item: item[0])

    checkpoints: list[tuple[str, Path]] = [(str(step), path) for step, path in numbered]
    final = adapter_dir / "adapters.safetensors"
    if final.exists():
        checkpoints.append(("final", final))
    return checkpoints


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot LoRA weight-delta norm trends across training checkpoints."
    )
    parser.add_argument("--adapter-dir", type=Path, default=Path("out/sft"))
    parser.add_argument("--out", type=Path, default=Path("out/lora_weight_trends.png"))
    args = parser.parse_args(argv)

    checkpoints = discover_checkpoints(args.adapter_dir)
    if len(checkpoints) < 2:
        raise SystemExit(
            f"Need at least 2 checkpoints in {args.adapter_dir} to plot a trend, "
            f"found {len(checkpoints)}"
        )

    labels = [label for label, _ in checkpoints]
    per_checkpoint_layer_norms = [
        aggregate_by_layer(load_lora_deltas(path)) for _, path in checkpoints
    ]
    plot_norm_trends(labels, per_checkpoint_layer_norms, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

Move the `import argparse` line up to the top of the file with the other stdlib imports (`re`,
`pathlib`) to satisfy the ruff `I` (isort) rule.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_weights_viz.py -v`
Expected: 6 passed

- [ ] **Step 5: Manual end-to-end smoke test against the real checkpoints**

Run:

```bash
uv run python -m autotune.weights_viz --adapter-dir out/sft --out out/lora_weight_trends.png
```

Expected: prints `Wrote out/lora_weight_trends.png`; the file exists
(`ls -la out/lora_weight_trends.png` shows size > 0). Open it and confirm 16 lines (one per
layer), x-axis ticks `100, 200, 300, final`.

- [ ] **Step 6: Lint and final commit**

Run: `uv run ruff check autotune/weights_viz.py tests/test_weights_viz.py`
Expected: no errors.

```bash
git add autotune/weights_viz.py tests/test_weights_viz.py
git commit -m "feat: add checkpoint discovery and CLI for LoRA weight trend plotting"
```
