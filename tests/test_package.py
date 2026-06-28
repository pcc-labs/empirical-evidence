from autotune.package import build_manifest, hash_bytes


def test_hash_bytes_is_stable_and_short():
    h1 = hash_bytes(b"the same content")
    h2 = hash_bytes(b"the same content")
    assert h1 == h2 and len(h1) == 16
    assert hash_bytes(b"different") != h1


def test_build_manifest_records_what_shipped():
    m = build_manifest(
        base_model="EricFillion/smollm3-3b-mlx",
        adapter_path="/abs/out/sft",
        fused_path="/abs/out/package/q4",
        iters=300,
        seed_states=["/abs/states/route1.state", "/abs/states/first.state"],
        data_sha="deadbeefdeadbeef",
        eval_summary={"passed": True, "proposer_reward": 6.0, "heuristic_reward": 5.0, "n": 2},
        quant_bits=4,
        stamp="2026-06-28T00:00:00Z",
    )
    assert m["base_model"] == "EricFillion/smollm3-3b-mlx"
    assert m["quantized_bits"] == 4
    assert m["seed_states"] == ["route1.state", "first.state"]  # basenames only
    assert m["eval"]["passed"] is True
    assert m["train_data_sha256_16"] == "deadbeefdeadbeef"
    assert m["created"] == "2026-06-28T00:00:00Z"


def test_build_manifest_allows_no_eval_and_no_quant():
    m = build_manifest(
        base_model="m",
        adapter_path="a",
        fused_path="f",
        iters=120,
        seed_states=[],
        data_sha="x",
        eval_summary=None,
        quant_bits=None,
        stamp="s",
    )
    assert m["eval"] is None and m["quantized_bits"] is None
