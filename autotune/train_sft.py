"""Train a LoRA adapter on the rejection-sampled SFT data.

Two backends behind one ``train(cfg, data_dir, iters) -> adapter_dir`` contract:

* ``cuda`` (default) — bf16 LoRA via TRL's ``SFTTrainer`` + PEFT, in-process on the GPU. Mirrors
  ``../overdub/train_sft.py`` but skips 4-bit (a 3B base is tiny on a 32 GB card).
* ``mlx`` — the Apple-Silicon path: build an ``mlx_lm.lora`` config from a :class:`MacProfile`
  and run ``mlx_lm lora`` as a subprocess.

This is the only place weights are updated; it is an IO/GPU wrapper exercised by the smoke run,
not unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from autotune.config import Config, load_config


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon)
# ---------------------------------------------------------------------------


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


def _train_mlx(  # pragma: no cover - subprocess driver, exercised by the smoke run
    cfg: Config, data_dir: Path, adapter_dir: Path, iters: int | None
) -> Path:
    """Run LoRA SFT via ``mlx_lm lora`` as a subprocess (Apple Silicon)."""
    config = build_lora_config(cfg, data_dir, adapter_dir, iters)
    config_path = write_config(config, cfg.storage.out_dir / "sft.lora.yaml")
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", str(config_path.resolve())]
    print(f"[train_sft] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return adapter_dir


# ---------------------------------------------------------------------------
# CUDA backend (HF/TRL/PEFT)
# ---------------------------------------------------------------------------


_THINKING_DEFAULT_BLOCK = (
    "{%- if enable_thinking is not defined %}\n"
    "    {%- set enable_thinking = false %}\n"
    "{%- endif %}\n"
)


def _patch_chat_template_disable_thinking(out_dir: Path) -> None:
    """Default SmolLM3's chat template to ``enable_thinking=false`` so the proposer emits flat JSON.

    Mirrors overdub: SmolLM3 is a hybrid-reasoning model that otherwise prepends ``<think>`` traces,
    which ``nudge_steer.parse_genome_response`` cannot read.
    """
    tpl = out_dir / "chat_template.jinja"
    if not tpl.exists():
        return
    text = tpl.read_text()
    if not text.startswith(_THINKING_DEFAULT_BLOCK):
        tpl.write_text(_THINKING_DEFAULT_BLOCK + text)


def _train_cuda(  # pragma: no cover - GPU driver, exercised by the smoke run
    cfg: Config, data_dir: Path, adapter_dir: Path, iters: int | None
) -> Path:
    """Train a bf16 LoRA adapter with TRL's SFTTrainer + PEFT. ``iters`` overrides max steps."""
    # Reduce CUDA memory fragmentation; must be set before torch imports.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    profile = cfg.profile
    rows = load_jsonl(Path(data_dir) / "train.jsonl")
    if not rows:
        raise RuntimeError(f"No SFT examples at {data_dir}/train.jsonl — run harvest first.")
    dataset = Dataset.from_list(rows)
    print(f"[train_sft] loaded {len(dataset)} SFT examples; base={cfg.model.base_model}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = get_peft_model(
        model,
        LoraConfig(
            r=profile.lora_r,
            lora_alpha=profile.lora_alpha,
            lora_dropout=profile.lora_dropout,
            target_modules=list(profile.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TRLSFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=profile.num_train_epochs,
        max_steps=iters if iters is not None else -1,
        per_device_train_batch_size=profile.per_device_train_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=profile.learning_rate,
        warmup_ratio=profile.warmup_ratio,
        lr_scheduler_type=profile.lr_scheduler_type,
        bf16=True,
        max_length=profile.max_seq_length,
        gradient_checkpointing=profile.gradient_checkpointing,
        logging_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer
    )
    print("[train_sft] starting CUDA LoRA SFT...")
    trainer.train()

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _patch_chat_template_disable_thinking(adapter_dir)
    return adapter_dir


def train(cfg: Config, data_dir: Path, iters: int | None = None) -> Path:
    """Run LoRA SFT on the active backend and return the adapter directory."""
    adapter_dir = cfg.storage.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if cfg.backend == "cuda":
        return _train_cuda(cfg, data_dir, adapter_dir, iters)
    return _train_mlx(cfg, data_dir, adapter_dir, iters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter (CUDA or MLX backend).")
    parser.add_argument("--data-dir", type=str, default=None, help="SFT data dir")
    parser.add_argument(
        "--iters", type=int, default=None, help="override training iters/max-steps"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    data_dir = Path(args.data_dir) if args.data_dir else cfg.storage.sft_dir
    print(f"[train_sft] backend={cfg.backend} profile={cfg.profile.name}")
    adapter = train(cfg, data_dir, iters=args.iters)
    print(f"[train_sft] adapter written to {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
