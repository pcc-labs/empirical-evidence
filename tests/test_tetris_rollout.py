import json

import pytest

from autotune.tetris_rollout import bench_command, mint, proxy_answers, proxy_command


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


def test_mint_refuses_to_start_a_game_when_the_proxy_does_not_answer(tmp_path):
    started = []

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_run(cmd, **kw):
        started.append(cmd)
        raise AssertionError("tetris-bench must not run")

    with pytest.raises(RuntimeError, match="proxy"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434", startup_s=0.0,
            run=fake_run, popen=lambda *a, **k: FakeProc(),
            answers=lambda url, timeout_s=2.0: False,
        )
    assert started == []


def test_mint_records_the_new_run_dirs_with_their_seed(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "older").mkdir()
    seen_env = {}

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        seen_env.update(env or {})
        seed = cmd[cmd.index("--seeds") + 1]
        d = runs / f"20260904-00000{seed}-abc"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"label": "pi/gemma4:26b/features/off+fixed"}))
        (d / "summary.json").write_text(json.dumps({"run_id": d.name, "fitness": {}}))

    rows = mint(
        [100, 101], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
        proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
        run=fake_run, popen=lambda *a, **k: FakeProc(), answers=lambda url, timeout_s=2.0: True,
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
    terminated = []

    class FakeProc:
        def terminate(self):
            terminated.append(True)

        def wait(self, timeout=None):
            pass

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        pass  # completes "successfully" but writes no run directory

    with pytest.raises(RuntimeError, match="100"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
            run=fake_run, popen=lambda *a, **k: FakeProc(), answers=lambda url, timeout_s=2.0: True,
        )
    assert terminated == [True]  # proxy still torn down on this raise path
