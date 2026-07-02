"""Benchmark each LoRA checkpoint on the forest crossing, split into per-domain sub-scores.

The map-grained ``benchmark`` saturates flat at 6.0 on route1/first_battle — no gradient and no
per-domain signal. This sweep instead drives the genome-driven forest follower (navigation fixed
along the canonical route; the genome varies survival) once per checkpoint per forest state:
each checkpoint's proposer is asked to improve the base genome given a shared baseline crossing,
the proposed genome is run, and the dense ``forest_story`` reward is reported as nav / battle /
discovery sub-scores — the behavioural trend to pair 1:1 with ``weights_viz``'s weight-movement
trend. Item beats (2, 5, 7) score 0 until pokemon-kafka emits ``bag_count`` (accepted).

``propose_forest_genome`` / ``summarize_verdicts`` / ``format_forest_trend`` are pure and
unit-tested; the emulator sweep, plot, and CLI are exercised by the smoke run (AGENTS.md).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from autotune.benchmark import discover_adapter_dirs
from autotune.config import Config, load_config
from autotune.forest_follow import ROUTE_DEFAULT, follow_once
from autotune.forest_story import DOMAINS, ForestVerdict, domain_scores, score_forest
from autotune.generate import make_proposer
from autotune.genome import base_genome
from autotune.harvest import resolve_states
from autotune.nudge_sft import _FOREST_SYSTEM, build_forest_mutation_prompt
from autotune.nudge_steer import parse_genome_response


@dataclass(frozen=True)
class ForestBenchRow:
    """One checkpoint's mean forest performance across the benchmark states."""

    label: str
    reward: float  # mean beats_passed
    domains: dict[str, float]  # mean per-domain sub-score
    crossed: float  # fraction of states crossed
    parsed: bool  # False if ANY state's proposer output was unparseable (fallback genome ran)
    turns: float = 0.0  # mean turns survived across states — finer-grained than the beat flags


def propose_forest_genome(
    proposer, params: dict, verdict: ForestVerdict
) -> tuple[dict, bool]:
    """Ask a checkpoint's proposer to improve ``params`` given the baseline ``verdict``.

    Returns ``(genome, parsed)``. Unparseable output falls back to ``params`` unchanged so the
    sweep still produces a row; the row is flagged instead of silently dropped.
    """
    text = proposer(build_forest_mutation_prompt(params, verdict))
    parsed = parse_genome_response(text)
    if parsed is None:
        return dict(params), False
    return {**params, **parsed}, True


def summarize_verdicts(
    label: str,
    verdicts: list[ForestVerdict],
    parsed: bool = True,
    turns: list[float] | None = None,
) -> ForestBenchRow:
    """Mean the dense reward and per-domain sub-scores across states."""
    n = max(1, len(verdicts))
    return ForestBenchRow(
        label=label,
        reward=sum(v.reward for v in verdicts) / n,
        domains={d: sum(domain_scores(v)[d] for v in verdicts) / n for d in DOMAINS},
        crossed=sum(1 for v in verdicts if v.crossed) / n,
        parsed=parsed,
        turns=(sum(turns) / max(1, len(turns))) if turns else 0.0,
    )


def format_forest_trend(baseline: ForestBenchRow, rows: list[ForestBenchRow]) -> str:
    """Per-checkpoint table: total forest reward + nav/battle/discovery sub-scores."""
    lines = [
        "forest benchmark: proposer genome per checkpoint vs base genome "
        f"(baseline reward {baseline.reward:.2f})",
        f"  {'checkpoint':>10}  {'reward':>7}  {'nav':>5}  {'battle':>6}  {'discov':>6}  "
        f"{'crossed':>7}  {'turns':>6}",
    ]
    for r in [baseline] + rows:
        flag = "" if r.parsed else "  (parse-fallback)"
        lines.append(
            f"  {r.label:>10}  {r.reward:>7.2f}  {r.domains['nav']:>5.2f}  "
            f"{r.domains['battle']:>6.2f}  {r.domains['discovery']:>6.2f}  "
            f"{r.crossed:>7.2f}  {r.turns:>6.0f}{flag}"
        )
    return "\n".join(lines)


def plot_forest_trend(  # pragma: no cover - plotting, mirrors weights_viz.plot_norm_trends
    baseline: ForestBenchRow, rows: list[ForestBenchRow], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r.label for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, [r.reward for r in rows], marker="o", linewidth=2.5, label="total reward")
    for domain in DOMAINS:
        ax.plot(labels, [r.domains[domain] for r in rows], marker="o", label=domain)
    ax.axhline(baseline.reward, linestyle="--", color="gray", label="base-genome baseline")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("mean forest sub-beats reached")
    ax.set_title("Forest reward by checkpoint, split by behavior domain")

    ax2 = ax.twinx()
    ax2.plot(labels, [r.turns for r in rows], marker="s", linestyle=":",
             color="tab:brown", label="turns survived")
    ax2.set_ylabel("mean turns survived")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_forest_benchmark(  # pragma: no cover - emulator sweep, exercised by the smoke run
    cfg: Config,
    adapter_dir: Path,
    state_paths: list[str],
    max_steps: int,
    worldmap_in: str | None,
) -> tuple[ForestBenchRow, list[ForestBenchRow]]:
    """Sweep every checkpoint's proposed genome through the forest follower, per state.

    The base-genome baseline run is adapter-independent, so it runs once per state and doubles
    as the situation the proposer is asked to improve.
    """
    checkpoints = discover_adapter_dirs(Path(adapter_dir))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {adapter_dir} to benchmark.")
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM.")
    rom = str(cfg.env.rom_path.resolve())
    route = ROUTE_DEFAULT if Path(ROUTE_DEFAULT).exists() else None

    baseline_verdicts: list[ForestVerdict] = []
    baseline_turns: list[float] = []
    per_ckpt: dict[str, list[ForestVerdict]] = {label: [] for label, _ in checkpoints}
    per_turns: dict[str, list[float]] = {label: [] for label, _ in checkpoints}
    parsed_ok: dict[str, bool] = {label: True for label, _ in checkpoints}

    baselines: list[tuple[str, ForestVerdict]] = []
    for state in state_paths:
        base_result = follow_once(
            rom, state, base_genome(), max_steps=max_steps, worldmap_in=worldmap_in, route=route
        )
        base_verdict = score_forest(base_result["events"])
        baseline_verdicts.append(base_verdict)
        baseline_turns.append(base_result["fitness"]["turns"])
        baselines.append((state, base_verdict))
        print(f"[forest-bench] {Path(state).name}: baseline reward={base_verdict.reward}")

    # Checkpoint-outer so each adapter is loaded exactly once: the model cache holds ONE
    # resident model, and cycling checkpoints inside the state loop would reload per pair
    # (and, before eviction existed, accumulated one 6 GiB model per checkpoint and OOMed).
    for label, ckpt_dir in checkpoints:
        proposer = make_proposer(cfg, ckpt_dir, system=_FOREST_SYSTEM)
        for state, base_verdict in baselines:
            genome, parsed = propose_forest_genome(proposer, base_genome(), base_verdict)
            parsed_ok[label] = parsed_ok[label] and parsed
            result = follow_once(
                rom, state, genome, max_steps=max_steps, worldmap_in=worldmap_in, route=route
            )
            verdict = score_forest(result["events"])
            per_ckpt[label].append(verdict)
            per_turns[label].append(result["fitness"]["turns"])
            print(
                f"[forest-bench] {Path(state).name}: checkpoint {label} "
                f"reward={verdict.reward} domains={domain_scores(verdict)} parsed={parsed}"
            )

    baseline = summarize_verdicts("baseline", baseline_verdicts, turns=baseline_turns)
    rows = [
        summarize_verdicts(label, per_ckpt[label], parsed_ok[label], turns=per_turns[label])
        for label, _ in checkpoints
    ]
    return baseline, rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Benchmark each LoRA checkpoint on the forest crossing, per behavior domain."
    )
    p.add_argument("--adapter-dir", type=Path, default=Path("out/sft"))
    p.add_argument("--states", default="states/forest_lv", help="a .state file or a dir of them")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--worldmap-in", default="states/forest_follow_wm.json")
    p.add_argument("--out", type=Path, default=Path("out/forest_benchmark_trends.png"))
    args = p.parse_args(argv)

    cfg = load_config()
    state_paths = resolve_states(args.states)
    if not state_paths:
        print(f"No save states found at {args.states!r}.", file=sys.stderr)
        return 2
    wm = args.worldmap_in if args.worldmap_in and Path(args.worldmap_in).exists() else None

    baseline, rows = run_forest_benchmark(
        cfg, args.adapter_dir, state_paths, args.max_steps, wm
    )
    print(format_forest_trend(baseline, rows))
    plot_forest_trend(baseline, rows, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
