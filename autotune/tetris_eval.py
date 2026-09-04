"""Tier-2 gate: the live benchmark over held-out seeds, tuned against its own base.

Reads a `tetris-bench` result file (data/benchmarks/benchmark-*.json in the
tetris repo). Three rules, all required: median race beats the baseline's
median; tokens per decision <= 1.5x the baseline's; late decisions do not
increase. The goal (>= 530) is reported, not gated.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RATIO = 1.5
GOAL_RACE = 530.0


def race(run: dict) -> float:
    """tetris_agent.fitness.race_score, restated so this module needs no tetris import at eval
    time.
    """
    f = run.get("fitness") or {}
    return float(f.get("score", 0)) + 5.0 * float(f.get("pieces_placed", 0))


def _tokens(run: dict) -> float:
    return float(((run.get("fitness") or {}).get("policy") or {}).get("tokens_per_decision", 0.0))


def _late(run: dict) -> int:
    return int(((run.get("fitness") or {}).get("policy") or {}).get("late", 0))


def runs_for(result: dict, model: str) -> list[dict]:
    prefix = f"{model}/"
    return [
        r
        for r in result.get("runs", [])
        if str(r.get("arm", "")).startswith(prefix) and not r.get("error")
    ]


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    tuned_median: float = 0.0
    base_median: float = 0.0
    tuned_tokens: float = 0.0
    base_tokens: float = 0.0
    tuned_late: int = 0
    base_late: int = 0

    def to_dict(self) -> dict:
        return {**self.__dict__, "goal_race": GOAL_RACE, "goal_met": self.tuned_median >= GOAL_RACE}


def gate(tuned_runs: list[dict], base_runs: list[dict]) -> GateResult:
    if not tuned_runs or not base_runs:
        return GateResult(False, ["no runs for tuned or base arm"])
    r = GateResult(
        passed=True,
        tuned_median=statistics.median(race(x) for x in tuned_runs),
        base_median=statistics.median(race(x) for x in base_runs),
        tuned_tokens=statistics.fmean(_tokens(x) for x in tuned_runs),
        base_tokens=statistics.fmean(_tokens(x) for x in base_runs),
        tuned_late=sum(_late(x) for x in tuned_runs),
        base_late=sum(_late(x) for x in base_runs),
    )
    if not r.tuned_median > r.base_median:
        r.reasons.append(
            f"median race {r.tuned_median:.0f} does not beat baseline {r.base_median:.0f}"
        )
    if r.tuned_tokens > TOKEN_RATIO * r.base_tokens + 1e-9:
        r.reasons.append(
            f"tokens/decision {r.tuned_tokens:.1f} > {TOKEN_RATIO} x baseline {r.base_tokens:.1f}"
        )
    if r.tuned_late > r.base_late:
        r.reasons.append(f"late decisions rose {r.base_late} -> {r.tuned_late}")
    r.passed = not r.reasons
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tetris_eval", description="tier-2 gate over a tetris-bench result file"
    )
    ap.add_argument(
        "--results", required=True, help="data/benchmarks/benchmark-*.json from the tetris repo"
    )
    ap.add_argument("--tuned", required=True, help="pi/<tuned tag>")
    ap.add_argument("--base", default="pi/gemma4:latest")
    args = ap.parse_args(argv)
    result = json.loads(Path(args.results).read_text())
    verdict = gate(runs_for(result, args.tuned), runs_for(result, args.base))
    print(json.dumps(verdict.to_dict(), indent=2))
    print("PASS" if verdict.passed else "FAIL: " + "; ".join(verdict.reasons))
    return 0 if verdict.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
