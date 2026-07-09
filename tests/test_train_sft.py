"""Tests for train_sft's pure config builder (the mlx side; GPU drivers are smoke-tested)."""

from __future__ import annotations

from autotune.config import load_config
from autotune.train_sft import build_lora_config


def test_build_lora_config_save_every_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_BACKEND", "mlx")
    cfg = load_config()
    with_save = build_lora_config(cfg, tmp_path, tmp_path / "sft", iters=40, save_steps=10)
    assert with_save["save_every"] == 10
    without = build_lora_config(cfg, tmp_path, tmp_path / "sft", iters=40)
    assert "save_every" not in without


def test_build_lora_config_has_warmup_cosine_schedule(tmp_path, monkeypatch):
    # mlx_lm.lora applies learning_rate as a constant without lr_schedule, which diverges;
    # the config must always emit warmup + cosine decay whose decay window matches iters.
    monkeypatch.setenv("AUTOTUNE_BACKEND", "mlx")
    cfg = load_config()
    config = build_lora_config(cfg, tmp_path, tmp_path / "sft", iters=900)
    sched = config["lr_schedule"]
    assert sched["name"] == "cosine_decay"
    assert sched["warmup"] == cfg.profile.warmup_steps
    peak, decay_steps, end = sched["arguments"]
    assert peak == cfg.profile.learning_rate  # cosine peaks at the profile LR
    assert decay_steps == 900  # decay window tracks the resolved iters, not the default
    assert end == cfg.profile.lr_min
    assert 0.0 < end < peak
