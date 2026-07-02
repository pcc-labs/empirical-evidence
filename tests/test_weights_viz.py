import numpy as np
import pytest
from safetensors.numpy import save_file

from autotune.weights_viz import aggregate_by_layer, discover_checkpoints, load_lora_deltas


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


def test_discover_checkpoints_peft_layout(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-2" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    ckpts = discover_checkpoints(tmp_path)
    assert [label for label, _ in ckpts] == ["2", "10", "final"]
    assert ckpts[-1][1] == tmp_path / "adapter_model.safetensors"


def test_discover_checkpoints_mlx_layout_wins(tmp_path):
    (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-5" / "adapter_model.safetensors").write_bytes(b"")
    ckpts = discover_checkpoints(tmp_path)
    assert [label for label, _ in ckpts] == ["100", "final"]
