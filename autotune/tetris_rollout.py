"""Try: mint teacher games in tetris under a tapes capture proxy.

Owns the proxy for one invocation (project `tetris-mint-<timestamp>`), refuses to
start a game unless the proxy answers and the listen port was free beforehand,
runs `tetris-bench --paused` one seed at a time, verifies the first seed actually
landed rows in Postgres (a direct `SELECT count(*) FROM raw_turns`, not the tapes
read API mint never starts), and writes a manifest of the run directories it
produced. Subprocess wrapper; exercised by the smoke run, not unit tests, except
for the pure parts.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
# tetris/scripts/tapes-up.sh binds :8092 for its own Ollama proxy -- mint used to
# collide with it there. A busy :8092 makes `tapes serve proxy` fail to bind,
# straight into the stderr this module used to discard, and games would be
# captured under tapes-up's project instead of the mint's.
DEFAULT_PROXY_LISTEN = "127.0.0.1:8093"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5432
PROXY_STARTUP_S = 10.0
PROXY_LOG_TAIL_LINES = 20


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


def proxy_dsn() -> str:
    """The Postgres DSN `tapes serve proxy` writes captures to.

    `tapes serve proxy` exits immediately with `empty postgres dsn` unless it is
    given one: it does not read `TAPES_PG_DSN` itself, and `~/.tapes/config.toml`
    ships an empty `[storage]` section. Passing it explicitly is what makes the
    proxy start at all.
    """
    dsn = os.environ.get("TAPES_PG_DSN", "")
    if not dsn:
        raise RuntimeError(
            "TAPES_PG_DSN is unset — `tapes serve proxy` needs the Postgres DSN passed "
            "explicitly (it exits with 'empty postgres dsn' otherwise). Export it, e.g. "
            "TAPES_PG_DSN=postgres://tapes:tapes@127.0.0.1:5432/tapes?sslmode=disable"
        )
    return dsn


def proxy_command(project: str, listen: str, upstream: str, dsn: str | None = None) -> list[str]:
    return [
        "tapes", "serve", "proxy",
        "--provider", "openai",
        "--upstream", upstream,
        "--listen", listen,
        "--project", project,
        "--postgres", dsn if dsn is not None else proxy_dsn(),
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


def port_free(listen: str, timeout_s: float = 0.5, connect=socket.create_connection) -> bool:
    """True when nothing answers a TCP connect on `listen` (host:port).

    Checked before the proxy is started: `tapes serve proxy` fails to bind on a
    port something else already owns, silently into the stderr this module used
    to discard, and `proxy_answers` ends up satisfied by whatever else is
    listening there instead -- games get captured under the wrong project and
    nothing notices.
    """
    host, _, port_s = listen.partition(":")
    try:
        connect((host, int(port_s)), timeout=timeout_s).close()
        return False  # something answered -- the port is not free
    except OSError:
        return True


def turns_captured(
    dsn: str, since: str, timeout_s: float = 5.0, run=subprocess.run, which=shutil.which
) -> int:
    """Count of `raw_turns` rows tapes has written to Postgres since `since` (RFC3339).

    Queries Postgres directly rather than tapes' read API: `mint` starts only the
    capture proxy (`:8093`), never the read API (`:8081`), so a check that
    depended on the read API answering passed only when something else had
    started it earlier. Postgres is already a hard requirement of `mint`
    (`postgres_check` refuses without it) and `psql` is expected on this box.
    `provider = 'openai'` excludes the operator's own Claude Code turns, which
    land under `provider = 'anthropic'` or with no provider at all.
    """
    if which("psql") is None:
        raise RuntimeError(
            "psql is required to verify capture via Postgres and was not found on PATH"
        )
    query = (
        "SELECT count(*) FROM raw_turns WHERE provider = 'openai' "
        f"AND received_at > '{since}'"
    )
    proc = run(
        ["psql", dsn, "-t", "-A", "-c", query], capture_output=True, text=True, timeout=timeout_s
    )
    return int(proc.stdout.strip())


def _run_dirs(runs_dir: Path) -> set[str]:
    return {p.name for p in runs_dir.iterdir() if p.is_dir()} if runs_dir.is_dir() else set()


def _tail_lines(path: Path, n: int) -> str:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


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
    dsn: str | None = None,
    run=subprocess.run,
    popen=subprocess.Popen,
    answers=proxy_answers,
    postgres_check=postgres_up,
    port_free_check=port_free,
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

    if not port_free_check(proxy_listen):
        raise RuntimeError(
            f"{proxy_listen} is already bound — tapes-up.sh (or something else) is likely "
            "listening there, and `tapes serve proxy` will fail to bind against it. Stop "
            "the other listener, or pass a different --proxy-listen."
        )

    dsn = dsn if dsn is not None else proxy_dsn()

    # `runs/` is gitignored in tetris, so a fresh checkout has none until the
    # first tetris-bench creates it -- and both the proxy log and the manifest
    # are opened before the first game runs. Without this the first real mint
    # dies on FileNotFoundError.
    runs_dir.mkdir(parents=True, exist_ok=True)
    proxy_log_path = runs_dir / f"mint-{project}.proxy.log"
    proxy_log = open(proxy_log_path, "w")
    try:
        proc = popen(
            proxy_command(project, proxy_listen, upstream, dsn=dsn),
            stdout=subprocess.DEVNULL,
            stderr=proxy_log,
        )
    except Exception:
        proxy_log.close()
        raise
    try:
        deadline = time.monotonic() + startup_s
        while not answers(base_url) and time.monotonic() < deadline:
            time.sleep(0.25)
        if not answers(base_url):
            proxy_log.flush()
            tail = _tail_lines(proxy_log_path, PROXY_LOG_TAIL_LINES)
            detail = f"\nlast lines of {proxy_log_path}:\n{tail}" if tail else ""
            raise RuntimeError(
                f"tapes proxy at {base_url} does not answer — is Postgres up "
                f"(`tapes local up`) and Ollama at {upstream}? Refusing to play uncaptured."
                f"{detail}"
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
                    n = turns_query(dsn, mint_started)
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
        proxy_log.close()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - subprocess driver
    from autotune.tetris_corpus import TRAIN_SEED_MAX, TRAIN_SEED_MIN

    ap = argparse.ArgumentParser(
        prog="tetris_rollout", description="mint teacher games under tapes capture"
    )
    ap.add_argument("--tetris-dir", default="../tetris")
    ap.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help=f"train-pool seeds, {TRAIN_SEED_MIN}-{TRAIN_SEED_MAX}",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    bad = [s for s in args.seeds if not (TRAIN_SEED_MIN <= s <= TRAIN_SEED_MAX)]
    if bad:
        ap.error(
            f"seeds must be in [{TRAIN_SEED_MIN}, {TRAIN_SEED_MAX}] -- outside that range "
            f"they are either the evaluation pool or, above {TRAIN_SEED_MAX}, PyBoy's "
            f"`& 0xFF` DIV mask (pyboy/plugins/base_plugin.py) wraps them back into a seed "
            f"the eval pool or a lower train seed already owns: {bad}"
        )
    rows = mint(
        args.seeds, tetris_dir=Path(args.tetris_dir), model=args.model, max_pieces=args.max_pieces,
        project=args.project,
    )
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
