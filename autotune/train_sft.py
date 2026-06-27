"""Train a LoRA adapter on the rejection-sampled SFT data, locally via MLX-LM.

Builds an ``mlx_lm.lora`` config from the resolved :class:`~autotune.config.MacProfile` and runs
``mlx_lm lora`` as a subprocess. This is the only place weights are updated; it is an IO/subprocess
wrapper exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from autotune.config import Config, load_config


def build_lora_config(
    cfg: Config, data_dir: Path, adapter_dir: Path, iters: int | None = None
) -> dict:
    """Assemble an mlx_lm.lora config dict from the Mac profile + model config."""
    profile = cfg.profile
    return {
        "model": cfg.model.base_model,
        "train": True,
        "data": str(Path(data_dir).resolve()),
        "fine_tune_type": "lora",
        "num_layers": profile.num_layers,
        "batch_size": profile.batch_size,
        "iters": iters if iters is not None else profile.iters,
        "learning_rate": profile.learning_rate,
        "steps_per_report": 10,
        "steps_per_eval": max(10, (iters or profile.iters) // 4),
        "adapter_path": str(Path(adapter_dir).resolve()),
        "max_seq_length": profile.max_seq_length,
        "grad_checkpoint": profile.grad_checkpoint,
        "lora_parameters": {
            "rank": profile.lora_rank,
            "scale": profile.lora_scale,
            "dropout": profile.lora_dropout,
        },
    }


def write_config(config: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def train(cfg: Config, data_dir: Path, iters: int | None = None) -> Path:
    """Run LoRA SFT and return the adapter directory."""
    adapter_dir = cfg.storage.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = build_lora_config(cfg, data_dir, adapter_dir, iters)
    config_path = write_config(config, cfg.storage.out_dir / "sft.lora.yaml")

    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", str(config_path.resolve())]
    print(f"[train_sft] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter via MLX-LM.")
    parser.add_argument("--data-dir", type=str, default=None, help="MLX-LM data dir")
    parser.add_argument("--iters", type=int, default=None, help="override training iterations")
    args = parser.parse_args(argv)

    cfg = load_config()
    data_dir = Path(args.data_dir) if args.data_dir else cfg.storage.sft_dir
    adapter = train(cfg, data_dir, iters=args.iters)
    print(f"[train_sft] adapter written to {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
