"""Try: mint teacher games in tetris under a tapes capture proxy.

Owns the proxy for one invocation (project `tetris-mint-<timestamp>`), refuses to
start a game unless the proxy answers, runs `tetris-bench --paused` one seed at a
time, and writes a manifest of the run directories it produced. Subprocess
wrapper; exercised by the smoke run, not unit tests, except for the pure parts.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "pi/gemma4:26b"
DEFAULT_MAX_PIECES = 100
DEFAULT_PROXY_LISTEN = "127.0.0.1:8092"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_API_BASE = "http://127.0.0.1:8081"
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5432
PROXY_STARTUP_S = 10.0


def bench_command(
    seed: int, *, model: str, max_pieces: int, harness: str = "features", effort: str = "off"
) -> list[str]:
    return [
        "uv", "run", "tetris-bench",
        "--paused", "--fixed-effort", "--no-control", "--no-power",
        "--models", model,
        "--harnesses", harness,
        "--efforts", effort,
        "--seeds", str(seed),
        "--max-pieces", str(max_pieces),
    ]


def proxy_command(project: str, listen: str, upstream: str) -> list[str]:
    return [
        "tapes", "serve", "proxy",
        "--provider", "openai",
        "--upstream", upstream,
        "--listen", listen,
        "--project", project,
    ]


def proxy_answers(base_url: str, timeout_s: float = 2.0) -> bool:
    """True when the proxy forwards a GET /v1/models to Ollama and gets a 200 back.

    This proves forwarding, not recording: the proxy answers whether or not
    tapes can write to Postgres. Pair with postgres_up before starting the
    proxy and turns_captured after the first seed to prove capture, not just
    forwarding.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def postgres_up(
    host: str = POSTGRES_HOST,
    port: int = POSTGRES_PORT,
    timeout_s: float = 2.0,
    connect=socket.create_connection,
) -> bool:
    """True when something answers a TCP connect on host:port.

    Checked before the proxy is even started: a proxy that forwards fine but
    has no Postgres behind it is exactly the configuration this module exists
    to prevent.
    """
    try:
        connect((host, port), timeout=timeout_s).close()
        return True
    except OSError:
        return False


def turns_captured(api_base: str, since: str, timeout_s: float = 5.0) -> int:
    """Turn count tapes has recorded since `since` (RFC3339), from its /v1/stats.

    Proof the proxy is not just forwarding but tapes is actually writing to
    Postgres: any failure to reach the read API counts as zero, the same as a
    reachable API reporting nothing captured.
    """
    try:
        with urllib.request.urlopen(f"{api_base}/v1/stats?since={since}", timeout=timeout_s) as r:
            return int(json.loads(r.read())["turn_count"])
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return 0


def _run_dirs(runs_dir: Path) -> set[str]:
    return {p.name for p in runs_dir.iterdir() if p.is_dir()} if runs_dir.is_dir() else set()


def mint(
    seeds,
    *,
    tetris_dir: Path,
    model: str = DEFAULT_MODEL,
    max_pieces: int = DEFAULT_MAX_PIECES,
    project: str | None = None,
    proxy_listen: str = DEFAULT_PROXY_LISTEN,
    upstream: str = DEFAULT_UPSTREAM,
    startup_s: float = PROXY_STARTUP_S,
    api_base: str = DEFAULT_API_BASE,
    run=subprocess.run,
    popen=subprocess.Popen,
    answers=proxy_answers,
    postgres_check=postgres_up,
    turns_query=turns_captured,
) -> list[dict]:
    tetris_dir = Path(tetris_dir)
    runs_dir = tetris_dir / "runs"
    project = project or f"tetris-mint-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    base_url = f"http://{proxy_listen}"

    if not postgres_check():
        raise RuntimeError(
            f"Postgres is not reachable at {POSTGRES_HOST}:{POSTGRES_PORT} — run "
            "`tapes local up` first. Refusing to start the capture proxy uncaptured."
        )

    proc = popen(
        proxy_command(project, proxy_listen, upstream),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + startup_s
        while not answers(base_url) and time.monotonic() < deadline:
            time.sleep(0.25)
        if not answers(base_url):
            raise RuntimeError(
                f"tapes proxy at {base_url} does not answer — is Postgres up "
                f"(`tapes local up`) and Ollama at {upstream}? Refusing to play uncaptured."
            )

        env = {**os.environ, "TETRIS_TAPES_OLLAMA_URL": f"{base_url}/v1"}
        mint_started = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        rows: list[dict] = []
        manifest_path = runs_dir / f"mint-{project}.jsonl"
        with open(manifest_path, "w") as manifest:
            for i, seed in enumerate(seeds):
                before = _run_dirs(runs_dir)
                run(
                    bench_command(seed, model=model, max_pieces=max_pieces),
                    cwd=str(tetris_dir), env=env, check=True,
                )
                new_dirs = sorted(_run_dirs(runs_dir) - before)
                if not new_dirs:
                    raise RuntimeError(
                        f"seed {seed} completed without recording a run — no new run directory "
                        f"appeared under {runs_dir}. Refusing to lose it silently."
                    )
                for name in new_dirs:
                    meta_path = runs_dir / name / "meta.json"
                    label = (
                        json.loads(meta_path.read_text()).get("label", "")
                        if meta_path.is_file() else ""
                    )
                    row = {"run_id": name, "seed": int(seed), "arm": label, "project": project}
                    rows.append(row)
                    # Written and flushed per seed, not once at the end: over a
                    # 60-seed mint this manifest is the only record of which
                    # seeds already ran, and an abort must not lose it.
                    manifest.write(json.dumps(row) + "\n")
                    manifest.flush()

                if i == 0:
                    # The proxy answering GET /v1/models only proves it forwards
                    # to Ollama, not that tapes can write to Postgres. Prove
                    # capture before spending the remaining seeds uncaptured.
                    n = turns_query(api_base, mint_started)
                    if n == 0:
                        raise RuntimeError(
                            f"tapes recorded zero turns since {mint_started} after seed "
                            f"{seed} completed — the proxy forwards but captures are not "
                            f"reaching Postgres. Refusing to play the remaining "
                            f"{len(seeds) - 1} seed(s) uncaptured."
                        )
        return rows
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - subprocess driver
    ap = argparse.ArgumentParser(
        prog="tetris_rollout", description="mint teacher games under tapes capture"
    )
    ap.add_argument("--tetris-dir", default="../tetris")
    ap.add_argument("--seeds", nargs="+", type=int, required=True, help="train-pool seeds, >= 100")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    low = [s for s in args.seeds if s < 100]
    if low:
        ap.error(f"seeds below 100 are the evaluation pool and are never minted: {low}")
    rows = mint(
        args.seeds, tetris_dir=Path(args.tetris_dir), model=args.model, max_pieces=args.max_pieces,
        project=args.project,
    )
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
