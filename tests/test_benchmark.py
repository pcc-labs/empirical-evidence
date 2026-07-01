from autotune.benchmark import format_trend, stage_checkpoint
from autotune.eval_proposer import EvalReport


def _adapter_dir(tmp_path):
    """A nested adapter dir (like ``out/sft``) so staging lands in a sibling ``bench_adapters``."""
    d = tmp_path / "sft"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"rank": 16}')
    return d


def test_stage_checkpoint_final_returns_adapter_dir_unchanged(tmp_path):
    adapter_dir = _adapter_dir(tmp_path)
    final = adapter_dir / "adapters.safetensors"
    assert stage_checkpoint(adapter_dir, "final", final) == adapter_dir


def test_stage_checkpoint_numbered_stages_config_and_weights(tmp_path):
    adapter_dir = _adapter_dir(tmp_path)
    ckpt = adapter_dir / "0000100_adapters.safetensors"
    ckpt.write_bytes(b"checkpoint-100-weights")

    staged = stage_checkpoint(adapter_dir, "100", ckpt)

    assert staged == tmp_path / "bench_adapters" / "ckpt-100"
    assert (staged / "adapter_config.json").read_text() == '{"rank": 16}'
    # The weights resolve to the checkpoint's bytes regardless of copy-vs-symlink.
    assert (staged / "adapters.safetensors").read_bytes() == b"checkpoint-100-weights"


def test_stage_checkpoint_is_idempotent(tmp_path):
    adapter_dir = _adapter_dir(tmp_path)
    ckpt = adapter_dir / "0000200_adapters.safetensors"
    ckpt.write_bytes(b"w200")

    stage_checkpoint(adapter_dir, "200", ckpt)
    staged = stage_checkpoint(adapter_dir, "200", ckpt)  # second call must not raise

    assert (staged / "adapters.safetensors").read_bytes() == b"w200"


def _report(reward, heuristic, *, passed, maps=2.0, turns=400):
    return EvalReport(1, reward, heuristic, turns, 500, maps, 2.0, passed)


def test_format_trend_lists_checkpoints_in_order_with_verdicts():
    reports = [
        ("100", _report(3.0, 5.0, passed=False)),
        ("200", _report(5.0, 5.0, passed=True, turns=450)),
        ("final", _report(6.0, 5.0, passed=True)),
    ]

    text = format_trend(reports)
    lines = text.splitlines()

    assert "heuristic (reward 5.00)" in lines[0]
    # Checkpoint rows preserve discover order; verdicts match each report.
    assert lines[2].split()[0] == "100" and lines[2].endswith("FAIL")
    assert lines[3].split()[0] == "200" and lines[3].endswith("PASS")
    assert lines[4].split()[0] == "final" and lines[4].endswith("PASS")


def test_format_trend_empty():
    assert format_trend([]) == "no checkpoints to benchmark"
