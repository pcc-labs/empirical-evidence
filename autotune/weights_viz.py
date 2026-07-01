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
