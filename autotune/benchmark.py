"""Benchmark each LoRA checkpoint behaviourally (the counterpart to ``weights_viz``).

``weights_viz`` plots *how far* the weights moved per checkpoint; a big norm can be a good or a bad
change. This asks the question that actually matters: *is the checkpoint any good?* It reuses the
exact gate ``eval_proposer`` runs — from each seed state, take a baseline rollout, let the trained
proposer and the deterministic heuristic each propose one genome, run both, and compare story
progress — but sweeps it across **every** checkpoint (100/200/300/final) so you get a
story-reward-vs-checkpoint trend to pair 1:1 with the norm-trend plot.

The heuristic and baseline rollouts don't depend on the adapter, so they're run **once per state**
and shared across checkpoints; only the proposer rollout varies per checkpoint.

``stage_checkpoint`` / ``format_trend`` are pure and unit-tested; the rollout sweep and CLI are
exercised by the smoke run (same split as ``eval_proposer``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.eval_proposer import EvalReport, _run_from, summarize_eval
from autotune.generate import make_proposer
from autotune.genome import base_genome
from autotune.nudge_steer import propose_next_genome
from autotune.story import load_story
from autotune.verifier import RolloutVerdict
from autotune.weights_viz import discover_checkpoints


def stage_checkpoint(adapter_dir: Path, label: str, ckpt_path: Path) -> Path:
    """Return an adapter dir that loads ``ckpt_path``'s weights.

    ``"final"`` is already a valid adapter dir, so it's returned as-is. A numbered checkpoint is a
    bare ``NNNN_adapters.safetensors`` file, so it's staged into
    ``<out>/bench_adapters/ckpt-<label>/`` as ``{adapter_config.json (copied),
    adapters.safetensors (symlinked)}`` — the shape both the mlx and PEFT proposer backends load.
    """
    if label == "final":
        return adapter_dir

    staged = adapter_dir.parent / "bench_adapters" / f"ckpt-{label}"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(adapter_dir / "adapter_config.json", staged / "adapter_config.json")

    weights = staged / "adapters.safetensors"
    if weights.exists() or weights.is_symlink():
        weights.unlink()
    os.symlink(ckpt_path.resolve(), weights)
    return staged


def format_trend(reports: list[tuple[str, EvalReport]]) -> str:
    """A compact table of proposer story reward per checkpoint against the shared heuristic."""
    if not reports:
        return "no checkpoints to benchmark"

    first = reports[0][1]
    lines = [
        f"benchmark: proposer per checkpoint vs heuristic "
        f"(reward {first.heuristic_reward:.2f}) over {first.n} state(s)",
        f"  {'checkpoint':>10}  {'reward':>7}  {'maps':>5}  {'turns':>6}  verdict",
    ]
    for label, r in reports:
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"  {label:>10}  {r.proposer_reward:>7.2f}  {r.proposer_maps:>5.2f}  "
            f"{r.proposer_turns:>6.0f}  {verdict}"
        )
    return "\n".join(lines)


def plot_benchmark_trend(  # pragma: no cover - plotting, mirrors weights_viz.plot_norm_trends
    labels: list[str],
    rewards: list[float],
    heuristic_reward: float,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, rewards, marker="o", label="proposer")
    ax.axhline(heuristic_reward, linestyle="--", color="gray", label="heuristic baseline")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("mean story reward")
    ax.set_title("LoRA proposer story reward by checkpoint")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_benchmark(  # pragma: no cover - subprocess/rollout sweep, exercised by the smoke run
    cfg: Config,
    adapter_dir: Path,
    state_paths: list[str],
    max_turns: int,
    seed: int,
) -> list[tuple[str, EvalReport]]:
    """Sweep the proposer-vs-heuristic eval across every checkpoint in ``adapter_dir``.

    Baseline and heuristic rollouts are adapter-independent, so they run once per state and are
    shared; only the proposer genome (and its rollout) is recomputed per checkpoint.
    """
    checkpoints = discover_checkpoints(adapter_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {adapter_dir} to benchmark.")

    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    heuristic_verdicts: list[RolloutVerdict] = []
    proposer_verdicts: dict[str, list[RolloutVerdict]] = {label: [] for label, _ in checkpoints}

    for state in state_paths:
        baseline = _run_from(cfg, base_genome(), state, "baseline", max_turns)
        h_genome = propose_next_genome(base_genome(), baseline, story, proposer=None, seed=seed)
        heuristic_verdicts.append(_run_from(cfg, h_genome, state, "heuristic", max_turns))
        print(f"[bench] {Path(state).name}: baseline reward={baseline.story_reward}")

        for label, ckpt_path in checkpoints:
            staged = stage_checkpoint(adapter_dir, label, ckpt_path)
            proposer = make_proposer(cfg, staged)
            p_genome = propose_next_genome(
                base_genome(), baseline, story, proposer=proposer, seed=seed
            )
            verdict = _run_from(cfg, p_genome, state, f"proposer-{label}", max_turns)
            proposer_verdicts[label].append(verdict)
            print(f"[bench] {Path(state).name}: checkpoint {label} reward={verdict.story_reward}")

    return [
        (label, summarize_eval(proposer_verdicts[label], heuristic_verdicts))
        for label, _ in checkpoints
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Benchmark each LoRA checkpoint's proposer against the heuristic baseline."
    )
    p.add_argument("--adapter-dir", type=Path, default=Path("out/sft"))
    p.add_argument("--states", default="states/", help="a .state file or a dir of them")
    p.add_argument("--max-turns", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("out/lora_benchmark_trends.png"))
    args = p.parse_args(argv)

    from autotune.harvest import resolve_states

    cfg = load_config()
    state_paths = resolve_states(args.states)
    if not state_paths:
        print(f"No save states found at {args.states!r}.", file=sys.stderr)
        return 2

    reports = run_benchmark(cfg, args.adapter_dir, state_paths, args.max_turns, args.seed)
    print(format_trend(reports))

    labels = [label for label, _ in reports]
    rewards = [r.proposer_reward for _, r in reports]
    heuristic_reward = reports[0][1].heuristic_reward if reports else 0.0
    plot_benchmark_trend(labels, rewards, heuristic_reward, args.out)
    print(f"Wrote {args.out}")

    return 0 if reports and all(r.passed for _, r in reports) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
