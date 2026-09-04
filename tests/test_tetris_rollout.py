import json

import pytest

from autotune.tetris_rollout import (
    bench_command,
    mint,
    postgres_up,
    proxy_answers,
    proxy_command,
    turns_captured,
)


def test_bench_command_is_the_spec_invocation():
    cmd = bench_command(101, model="pi/gemma4:26b", max_pieces=100)
    assert cmd[:3] == ["uv", "run", "tetris-bench"]
    for flag in ("--paused", "--fixed-effort", "--no-control", "--no-power"):
        assert flag in cmd
    assert cmd[cmd.index("--models") + 1] == "pi/gemma4:26b"
    assert cmd[cmd.index("--harnesses") + 1] == "features"
    assert cmd[cmd.index("--efforts") + 1] == "off"
    assert cmd[cmd.index("--seeds") + 1] == "101"
    assert cmd[cmd.index("--max-pieces") + 1] == "100"


def test_proxy_command_names_the_project():
    cmd = proxy_command("tetris-mint-x", "127.0.0.1:8092", "http://127.0.0.1:11434")
    assert cmd[:3] == ["tapes", "serve", "proxy"]
    assert cmd[cmd.index("--project") + 1] == "tetris-mint-x"
    assert cmd[cmd.index("--provider") + 1] == "openai"


def test_proxy_answers_is_false_when_nothing_listens():
    assert proxy_answers("http://127.0.0.1:1", timeout_s=0.2) is False


def test_postgres_up_is_false_when_nothing_listens():
    assert postgres_up("127.0.0.1", 1, timeout_s=0.2) is False


def test_turns_captured_is_zero_when_the_api_is_unreachable():
    assert turns_captured("http://127.0.0.1:1", "2026-09-04T00:00:00Z", timeout_s=0.2) == 0


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        pass


def _write_run_dir(runs, seed):
    d = runs / f"20260904-00000{seed}-abc"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"label": "pi/gemma4:26b/features/off+fixed"}))
    (d / "summary.json").write_text(json.dumps({"run_id": d.name, "fitness": {}}))
    return d


def test_mint_refuses_to_start_a_game_when_the_proxy_does_not_answer(tmp_path):
    started = []

    def fake_run(cmd, **kw):
        started.append(cmd)
        raise AssertionError("tetris-bench must not run")

    with pytest.raises(RuntimeError, match="proxy"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434", startup_s=0.0,
            run=fake_run, popen=lambda *a, **k: _FakeProc(),
            answers=lambda url, timeout_s=2.0: False,
            postgres_check=lambda: True, turns_query=lambda *a, **k: 1,
        )
    assert started == []


def test_mint_refuses_to_start_the_proxy_when_postgres_is_unreachable(tmp_path):
    popen_calls = []

    def fake_popen(*a, **kw):
        popen_calls.append(a)
        return _FakeProc()

    with pytest.raises(RuntimeError, match="tapes local up"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434", startup_s=0.0,
            run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
            popen=fake_popen, answers=lambda url, timeout_s=2.0: True,
            postgres_check=lambda: False, turns_query=lambda *a, **k: 1,
        )
    assert popen_calls == []  # the proxy itself must never start


def test_mint_refuses_when_the_first_seed_captures_zero_turns(tmp_path):
    """The proxy answering GET /v1/models only proves it forwards to Ollama, not
    that tapes can write to Postgres. Zero turns captured after a completed seed
    is the tell -- refuse to play the rest uncaptured."""
    runs = tmp_path / "runs"
    runs.mkdir()
    proc = _FakeProc()
    run_calls = []

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        run_calls.append(cmd)
        seed = cmd[cmd.index("--seeds") + 1]
        _write_run_dir(runs, seed)

    with pytest.raises(RuntimeError, match="zero turns"):
        mint(
            [100, 101], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
            run=fake_run, popen=lambda *a, **k: proc, answers=lambda url, timeout_s=2.0: True,
            postgres_check=lambda: True, turns_query=lambda *a, **k: 0,
        )
    assert len(run_calls) == 1  # seed 101 was never attempted
    assert proc.terminated  # proxy still torn down on this raise path


def test_mint_records_the_new_run_dirs_with_their_seed(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "older").mkdir()
    seen_env = {}

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        seen_env.update(env or {})
        seed = cmd[cmd.index("--seeds") + 1]
        _write_run_dir(runs, seed)

    rows = mint(
        [100, 101], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
        proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
        run=fake_run, popen=lambda *a, **k: _FakeProc(), answers=lambda url, timeout_s=2.0: True,
        postgres_check=lambda: True, turns_query=lambda *a, **k: 3,
    )
    assert [(r["seed"], r["arm"], r["project"]) for r in rows] == [
        (100, "pi/gemma4:26b/features/off+fixed", "p"),
        (101, "pi/gemma4:26b/features/off+fixed", "p"),
    ]
    assert seen_env["TETRIS_TAPES_OLLAMA_URL"] == "http://127.0.0.1:8092/v1"
    manifest = [json.loads(x) for x in (runs / "mint-p.jsonl").read_text().splitlines()]
    assert manifest == rows


def test_mint_refuses_a_seed_that_completes_without_a_new_run_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    proc = _FakeProc()

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        pass  # completes "successfully" but writes no run directory

    with pytest.raises(RuntimeError, match="100"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
            run=fake_run, popen=lambda *a, **k: proc, answers=lambda url, timeout_s=2.0: True,
            postgres_check=lambda: True, turns_query=lambda *a, **k: 3,
        )
    assert proc.terminated  # proxy still torn down on this raise path


def test_mint_writes_the_manifest_incrementally_surviving_a_later_raise(tmp_path):
    """Over a 60-seed, ~5-hour mint the manifest is the only record of which
    seeds already ran -- an abort on seed k must not lose seeds 1..k-1."""
    runs = tmp_path / "runs"
    runs.mkdir()

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        seed = cmd[cmd.index("--seeds") + 1]
        if seed == "101":
            return  # seed 101 completes without a run directory
        _write_run_dir(runs, seed)

    with pytest.raises(RuntimeError, match="101"):
        mint(
            [100, 101], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
            run=fake_run, popen=lambda *a, **k: _FakeProc(),
            answers=lambda url, timeout_s=2.0: True,
            postgres_check=lambda: True, turns_query=lambda *a, **k: 3,
        )
    manifest = [json.loads(x) for x in (runs / "mint-p.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in manifest] == [100]  # seed 100's row survived the raise


def test_mint_creates_the_runs_dir_when_the_checkout_has_none(tmp_path):
    """`runs/` is gitignored in tetris, so a fresh checkout has none until the
    first game writes one -- but the manifest is opened before the first game."""

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    runs = tmp_path / "runs"
    assert not runs.exists()

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        seed = cmd[cmd.index("--seeds") + 1]
        d = runs / f"20260904-00000{seed}-abc"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"label": "pi/gemma4:26b/features/off+fixed"}))
        (d / "summary.json").write_text(json.dumps({"run_id": d.name, "fitness": {}}))

    rows = mint(
        [100],
        tetris_dir=tmp_path,
        model="pi/gemma4:26b",
        max_pieces=10,
        project="p",
        proxy_listen="127.0.0.1:8092",
        upstream="http://127.0.0.1:11434",
        startup_s=0.0,
        run=fake_run,
        popen=lambda *a, **k: FakeProc(),
        answers=lambda url, timeout_s=2.0: True,
        postgres_check=lambda: True,
        turns_query=lambda *a, **k: 7,
    )
    assert [r["seed"] for r in rows] == [100]
    assert (runs / "mint-p.jsonl").is_file()
