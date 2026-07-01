import numpy as np
import pytest
from safetensors.numpy import save_file

from autotune.weights_viz import aggregate_by_layer, load_lora_deltas


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


def test_aggregate_by_layer_sums_within_layer_and_ignores_unmatched():
    deltas = {
        "layers.0.mlp.down_proj": 1.0,
        "layers.0.self_attn.q_proj": 2.0,
        "layers.1.mlp.up_proj": 5.0,
        "not_a_layer_key": 9.0,
    }

    assert aggregate_by_layer(deltas) == {0: 3.0, 1: 5.0}
