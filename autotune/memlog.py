"""Memory evidence for long-running converters.

The kernel OOM killer sends SIGKILL, and systemd-oomd then kills the whole terminal scope, so
nothing in-process gets to write a post-mortem. This logger writes one JSON line *per phase and
per input file, as it happens* (open/append/close each time), so the trail survives the kill. It
also enforces an RSS budget: past the limit the converter raises ``MemoryBudgetExceeded`` and
exits on its own, with the evidence, instead of taking the terminal down with it.

Stdlib only; reads ``/proc/self/status`` (falls back to ``resource`` where /proc is absent).
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import TextIO

GB = 1024**3
ENV_MAX_RSS_GB = "AUTOTUNE_MAX_RSS_GB"
DEFAULT_BUDGET_FRACTION = 0.5  # of MemTotal; well under both the kernel's and oomd's trip points


class MemoryBudgetExceeded(RuntimeError):
    def __init__(self, rss: int, limit: int, phase: str):
        self.rss, self.limit, self.phase = rss, limit, phase
        super().__init__(
            f"memory budget exceeded at {phase}: rss {rss / GB:.2f} GB > limit {limit / GB:.2f} GB"
        )


def _proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def rss_bytes() -> int:
    """Current resident set size."""
    kb = _proc_status_kb("VmRSS")
    if kb is not None:
        return kb * 1024
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def peak_rss_bytes() -> int:
    """High-water mark of the resident set size."""
    kb = _proc_status_kb("VmHWM")
    if kb is not None:
        return kb * 1024
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def mem_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def default_budget_bytes() -> int:
    """``AUTOTUNE_MAX_RSS_GB`` if set, else half of MemTotal, else 0 (unlimited)."""
    env = os.environ.get(ENV_MAX_RSS_GB)
    if env:
        return int(float(env) * GB)
    total = mem_total_bytes()
    return int(total * DEFAULT_BUDGET_FRACTION) if total else 0


class MemLog:
    """Append-only memory trail with an optional RSS budget.

    ``path=None`` keeps it in-process only; ``stream=None`` silences the console line. Every
    call to :meth:`log` records rss/peak/elapsed plus the caller's fields and raises
    :class:`MemoryBudgetExceeded` once rss passes ``limit_bytes`` (0 disables the check).
    """

    def __init__(
        self,
        path: Path | None = None,
        limit_bytes: int = 0,
        stream: TextIO | None = sys.stderr,
        label: str = "mem",
    ):
        self.path = path
        self.limit = int(limit_bytes)
        self.stream = stream
        self.label = label
        self.t0 = time.monotonic()
        self.count = 0
        self.kept_by_file: dict[str, int] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log(
            "start",
            check=False,
            pid=os.getpid(),
            limit_gb=round(self.limit / GB, 3),
            argv=sys.argv[1:],
        )

    def note_file(self, name: str, kept: int) -> None:
        """Remember how many events a file contributed, for the abort summary."""
        self.kept_by_file[name] = self.kept_by_file.get(name, 0) + kept

    def log(self, phase: str, check: bool = True, **fields) -> dict:
        """Record one phase; ``check=False`` records without enforcing the budget."""
        rss = rss_bytes()
        rec = {
            "t": round(time.monotonic() - self.t0, 3),
            "phase": phase,
            "rss_gb": round(rss / GB, 3),
            "peak_gb": round(peak_rss_bytes() / GB, 3),
            **fields,
        }
        self.count += 1
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        if self.stream is not None:
            extra = " ".join(f"{k}={v}" for k, v in fields.items() if k != "argv")
            print(
                f"[{self.label}] {phase} rss={rec['rss_gb']:.2f}GB peak={rec['peak_gb']:.2f}GB "
                f"t={rec['t']:.1f}s {extra}".rstrip(),
                file=self.stream,
                flush=True,
            )
        if check and self.limit and rss > self.limit:
            raise MemoryBudgetExceeded(rss, self.limit, phase)
        return rec

    def top_files(self, n: int = 5) -> list[tuple[str, int]]:
        return sorted(self.kept_by_file.items(), key=lambda kv: -kv[1])[:n]
