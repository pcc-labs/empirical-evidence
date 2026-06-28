"""Package the trained adapter into a standalone, versioned model — gated on the eval passing.

Fuses the LoRA adapter (``out/sft``) into the base model so the proposer can run without the
adapter side-loaded — PEFT ``merge_and_unload`` on CUDA, ``mlx_lm fuse`` on Apple Silicon (where it
can also quantize to fit the unified-memory budget) — and writes a ``manifest.json`` recording
exactly what was shipped (base model, data fingerprint, iters, seed states, eval verdict). The eval
gate is enforced first: a model that didn't beat the heuristic is not worth packaging, so
``--force`` is required to package a failing one.

``build_manifest`` / ``hash_bytes`` are pure and unit-tested; the fuse/quantize subprocess and CLI
are exercised by the smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config


def hash_bytes(data: bytes) -> str:
    """Stable short fingerprint of file content (first 16 hex of sha256)."""
    return hashlib.sha256(data).hexdigest()[:16]


def build_manifest(
    *,
    base_model: str,
    adapter_path: str,
    fused_path: str,
    iters: int,
    seed_states: list[str],
    data_sha: str,
    eval_summary: dict | None,
    quant_bits: int | None,
    stamp: str,
) -> dict:
    """Assemble the manifest dict recording what was packaged and how it scored. Pure."""
    return {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "fused_path": fused_path,
        "quantized_bits": quant_bits,
        "iters": iters,
        "seed_states": [Path(s).name for s in seed_states],
        "train_data_sha256_16": data_sha,
        "eval": eval_summary,
        "created": stamp,
    }


def _fuse_command(base_model: str, adapter_dir: Path, save_path: Path) -> list[str]:
    """mlx_lm fuse: merge the LoRA adapter into the base model."""
    return [
        "uv", "run", "python", "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", str(adapter_dir.resolve()),
        "--save-path", str(save_path.resolve()),
    ]


def _quantize_command(fused_path: Path, quant_path: Path, bits: int) -> list[str]:
    """mlx_lm convert -q: quantize the fused model to ``bits`` (e.g. 4 or 8)."""
    return [
        "uv", "run", "python", "-m", "mlx_lm", "convert",
        "--hf-path", str(fused_path.resolve()),
        "--mlx-path", str(quant_path.resolve()),
        "-q", "--q-bits", str(bits),
    ]


def _profile_iters(profile) -> int:
    """Training extent for the manifest, regardless of backend (MLX iters | CUDA epochs)."""
    return int(getattr(profile, "iters", getattr(profile, "num_train_epochs", 0)))


def _merge_cuda(cfg: Config, adapter_dir: Path, fused_path: Path) -> None:  # pragma: no cover
    """Merge the PEFT LoRA adapter into the base model and save it standalone (CUDA fuse)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir.resolve())).merge_and_unload()
    merged.save_pretrained(str(fused_path))
    AutoTokenizer.from_pretrained(str(adapter_dir.resolve())).save_pretrained(str(fused_path))


def run_package(  # pragma: no cover - fuse driver, exercised by the smoke run
    cfg: Config,
    *,
    quant_bits: int | None,
    eval_summary: dict | None,
    stamp: str,
    seed_states: list[str] | None = None,
) -> dict:
    """Fuse the adapter (+ optional MLX quantize) and write the manifest. Returns the manifest."""
    adapter_dir = cfg.storage.adapter_dir
    if not adapter_dir.exists():
        raise RuntimeError(f"No trained adapter at {adapter_dir} — run the harvest + train first.")

    out_root = cfg.storage.out_dir / "package"
    fused_path = out_root / "fused"
    fused_path.mkdir(parents=True, exist_ok=True)

    final_path = fused_path
    if cfg.backend == "cuda":
        _merge_cuda(cfg, adapter_dir, fused_path)
        if quant_bits:
            print(f"[package] ignoring --quantize q{quant_bits}: not needed on a 32 GB GPU.")
    else:
        subprocess.run(_fuse_command(cfg.model.base_model, adapter_dir, fused_path), check=True)
        if quant_bits:
            quant_path = out_root / f"q{quant_bits}"
            quant_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(_quantize_command(fused_path, quant_path, quant_bits), check=True)
            final_path = quant_path

    train_path = cfg.storage.sft_dir / "train.jsonl"
    data_sha = hash_bytes(train_path.read_bytes()) if train_path.exists() else "missing"

    manifest = build_manifest(
        base_model=cfg.model.base_model,
        adapter_path=str(adapter_dir.resolve()),
        fused_path=str(final_path.resolve()),
        iters=_profile_iters(cfg.profile),
        seed_states=seed_states or [],
        data_sha=data_sha,
        eval_summary=eval_summary,
        quant_bits=quant_bits if cfg.backend != "cuda" else None,
        stamp=stamp,
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    from datetime import datetime, timezone

    from autotune.eval_proposer import run_eval
    from autotune.harvest import resolve_states

    p = argparse.ArgumentParser(description="Package the trained proposer (gated on the eval).")
    p.add_argument("--states", default="states/", help="seed states for the eval gate")
    p.add_argument("--quantize", choices=["q4", "q8"], default=None)
    p.add_argument("--max-turns", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-eval", action="store_true", help="trust a prior eval; record none")
    p.add_argument("--force", action="store_true", help="package even if the eval fails")
    args = p.parse_args(argv)

    cfg = load_config()
    eval_summary = None
    state_paths = resolve_states(args.states)
    if not args.skip_eval:
        if not state_paths:
            print(f"No save states at {args.states!r} to run the eval gate.", file=sys.stderr)
            return 2
        report = run_eval(cfg, state_paths, args.max_turns, args.seed)
        eval_summary = {
            "passed": report.passed,
            "proposer_reward": report.proposer_reward,
            "heuristic_reward": report.heuristic_reward,
            "n": report.n,
        }
        if not report.passed and not args.force:
            print("[package] eval FAILED — proposer did not beat the heuristic. "
                  "Refusing to package (use --force to override).", file=sys.stderr)
            return 1

    quant_bits = {"q4": 4, "q8": 8}.get(args.quantize)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = run_package(
        cfg,
        quant_bits=quant_bits,
        eval_summary=eval_summary,
        stamp=stamp,
        seed_states=state_paths,
    )
    print(f"[package] wrote {manifest['fused_path']} + manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
