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
