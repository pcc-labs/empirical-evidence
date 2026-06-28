"""Eval: are the trained proposer weights actually worth anything? (the gate before packaging)

Behavioural, not token-match. From each seed state we take a baseline rollout (the default genome)
and its verdict, then let *both* the trained MLX proposer and the deterministic heuristic propose
one genome from that same baseline situation — exactly the ``propose_next_genome`` call the loop
makes. We run each proposed genome from the state and compare which advances further along the
story. Deterministic per ``(state, genome)`` ⇒ k=1.

Pass bar: the proposer matches or beats the heuristic on story progress at equal-or-fewer turns,
fully local (zero Claude calls). A fail here means the weights aren't worth packaging — which is
the point of running this first.

``summarize_eval`` / ``format_report`` are pure and unit-tested; the rollout driver and CLI are
exercised by the smoke run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.generate import make_proposer
from autotune.genome import base_genome
from autotune.nudge_steer import propose_next_genome
from autotune.rollout import run_batch
from autotune.story import load_story
from autotune.verifier import RolloutVerdict, verify


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _turns(v: RolloutVerdict) -> float:
    return float(v.fitness.get("turns", 0) or 0)


def _maps(v: RolloutVerdict) -> float:
    return float(v.fitness.get("maps_visited", 0) or 0)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate proposer-vs-heuristic comparison across the evaluated seed states."""

    n: int
    proposer_reward: float
    heuristic_reward: float
    proposer_turns: float
    heuristic_turns: float
    proposer_maps: float
    heuristic_maps: float
    passed: bool


def summarize_eval(
    proposer_verdicts: list[RolloutVerdict],
    heuristic_verdicts: list[RolloutVerdict],
) -> EvalReport:
    """Aggregate per-state verdicts and decide pass/fail.

    Pass = proposer's mean story progress is strictly higher, or equal with no more turns. With
    nothing evaluated (n=0) the result is a fail — there is no evidence the weights help.
    """
    n = min(len(proposer_verdicts), len(heuristic_verdicts))
    pv, hv = proposer_verdicts[:n], heuristic_verdicts[:n]
    p_reward, h_reward = _mean([v.story_reward for v in pv]), _mean([v.story_reward for v in hv])
    p_turns, h_turns = _mean([_turns(v) for v in pv]), _mean([_turns(v) for v in hv])

    if n == 0:
        passed = False
    elif p_reward > h_reward:
        passed = True
    elif p_reward == h_reward:
        passed = p_turns <= h_turns
    else:
        passed = False

    return EvalReport(
        n=n,
        proposer_reward=p_reward,
        heuristic_reward=h_reward,
        proposer_turns=p_turns,
        heuristic_turns=h_turns,
        proposer_maps=_mean([_maps(v) for v in pv]),
        heuristic_maps=_mean([_maps(v) for v in hv]),
        passed=passed,
    )


def format_report(report: EvalReport) -> str:
    """A compact, human-readable table of the comparison + the verdict."""
    verdict = "PASS" if report.passed else "FAIL"
    return (
        f"proposer vs heuristic over {report.n} state(s):\n"
        f"  story reward : {report.proposer_reward:.2f}  vs  {report.heuristic_reward:.2f}\n"
        f"  maps visited : {report.proposer_maps:.2f}  vs  {report.heuristic_maps:.2f}\n"
        f"  turns        : {report.proposer_turns:.0f}  vs  {report.heuristic_turns:.0f}\n"
        f"  verdict      : {verdict} "
        f"(proposer {'≥' if report.passed else '<'} heuristic on progress)"
    )


def _run_from(  # pragma: no cover - subprocess driver
    cfg: Config, genome: dict, state: str, label: str, max_turns: int
) -> RolloutVerdict:
    """Run one rollout of ``genome`` from ``state`` and return its verdict."""
    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    work_root = cfg.storage.out_dir / "eval" / Path(state).stem / label
    rollouts = run_batch(
        cfg,
        params_list=[dict(genome)],
        max_turns=max_turns,
        work_root=work_root,
        concurrency=1,
        load_state=state,
    )
    r = rollouts[0]
    return verify(story, r.fitness, r.events)


def run_eval(  # pragma: no cover - subprocess driver, exercised by the smoke run
    cfg: Config, state_paths: list[str], max_turns: int, seed: int
) -> EvalReport:
    """Compare the trained proposer against the heuristic, one genome each, per seed state."""
    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    proposer = make_proposer(cfg) if cfg.storage.adapter_dir.exists() else None
    if proposer is None:
        print("[eval] no trained adapter found — comparing heuristic against itself.")

    proposer_verdicts: list[RolloutVerdict] = []
    heuristic_verdicts: list[RolloutVerdict] = []
    for state in state_paths:
        baseline = _run_from(cfg, base_genome(), state, "baseline", max_turns)
        p_genome = propose_next_genome(base_genome(), baseline, story, proposer=proposer, seed=seed)
        h_genome = propose_next_genome(base_genome(), baseline, story, proposer=None, seed=seed)
        proposer_verdicts.append(_run_from(cfg, p_genome, state, "proposer", max_turns))
        heuristic_verdicts.append(_run_from(cfg, h_genome, state, "heuristic", max_turns))
        print(f"[eval] {Path(state).name}: baseline reward={baseline.story_reward}")

    return summarize_eval(proposer_verdicts, heuristic_verdicts)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    p = argparse.ArgumentParser(description="Eval the trained proposer vs the heuristic baseline.")
    p.add_argument("--states", default="states/", help="a .state file or a dir of them")
    p.add_argument("--max-turns", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from autotune.harvest import resolve_states

    cfg = load_config()
    state_paths = resolve_states(args.states)
    if not state_paths:
        print(f"No save states found at {args.states!r}.", file=sys.stderr)
        return 2

    report = run_eval(cfg, state_paths, args.max_turns, args.seed)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
