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
