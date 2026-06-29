"""Harvest the forest SFT buffer — the capture→dataset half of "reach Pewter, capture weights".

Sweeps a set of survival genomes through the genome-driven follower (:mod:`autotune.forest_follow`),
scores each crossing with the dense :mod:`autotune.forest_story` reward, and rejection-samples the
crossings into forest-keyed improvement pairs (:func:`autotune.nudge_sft.assemble_forest_corpus`)
written as ``train.jsonl`` / ``valid.jsonl`` for ``train_sft``.

The follower fixes navigation; the genome varies survival, so the sweep produces crossings of
different reward — the spread that gives the LoRA a gradient to climb. Runs are sequential: PyBoy
is one emulator per process, so we run one genome at a time rather than the parallel ``run_batch``.

IO/emulator driver: smoke-tested via a real sweep, not unit-tested (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import load_config
from autotune.forest_follow import follow_once
from autotune.forest_story import score_forest
from autotune.genome import base_genome
from autotune.nudge_sft import ForestWinner, assemble_forest_corpus, write_sft_data


def sweep_genomes(run_thresholds: list[float], heal_thresholds: list[float]) -> list[dict]:
    """Build the survival-genome grid to sweep (the levers that move the forest reward)."""
    genomes = []
    for run_thr in run_thresholds:
        for heal_thr in heal_thresholds:
            genomes.append({**base_genome(),
                            "hp_run_threshold": run_thr, "hp_heal_threshold": heal_thr})
    return genomes


def harvest(
    cfg,
    in_state: str,
    state_label: str,
    genomes: list[dict],
    max_steps: int,
    worldmap_in: str | None,
    out_dir: Path,
) -> dict:
    """Run each genome through the follower, score it, and write the forest SFT buffer.

    Returns a summary; writes ``leaderboard.json`` (every genome → reward) and the SFT data.
    """
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM.")
    rom = str(cfg.env.rom_path.resolve())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    winners: list[ForestWinner] = []
    board: list[dict] = []
    for i, genome in enumerate(genomes):
        result = follow_once(rom, in_state, genome, max_steps=max_steps, worldmap_in=worldmap_in)
        verdict = score_forest(result["events"])
        winners.append(ForestWinner(params=genome, verdict=verdict, fitness=result["fitness"]))
        row = {
            "genome_diffs": {k: genome[k] for k in ("hp_run_threshold", "hp_heal_threshold")},
            "reward": verdict.reward,
            "furthest": verdict.furthest_beat_name,
            "crossed": verdict.crossed,
            "trainer_wins": verdict.signals.trainer_wins,
            "turns": result["fitness"]["turns"],
        }
        board.append(row)
        print(f"[harvest] {i + 1}/{len(genomes)} {json.dumps(row)}", flush=True)

    board.sort(key=lambda r: (r["reward"], -r["turns"]), reverse=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2) + "\n")

    examples = assemble_forest_corpus({state_label: winners})
    rewards = sorted({w.verdict.reward for w in winners})
    if not examples:
        print(f"[harvest] no gradient (rewards seen: {rewards}) — buffer not written. "
              "Widen the sweep or raise max-steps so crossings differ.")
        crossed_any = any(w.verdict.crossed for w in winners)
        return {"examples": 0, "rewards_seen": rewards, "crossed_any": crossed_any}

    train_path, valid_path = write_sft_data(out_dir, examples)
    print(f"[harvest] {len(examples)} SFT pairs from {len(winners)} runs "
          f"(rewards seen: {rewards}) -> {train_path}")
    return {
        "examples": len(examples),
        "rewards_seen": rewards,
        "crossed_any": any(w.verdict.crossed for w in winners),
        "train": str(train_path),
        "valid": str(valid_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest forest SFT buffer from a genome sweep.")
    parser.add_argument("--in-state", default="states/forest_lv/lead_lv13_potions.state")
    parser.add_argument("--state-label", default="lv13", help="Group key for within-state pairs.")
    parser.add_argument("--worldmap-in", default="states/forest_follow_wm.json")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--out-dir", default="out/forest_sft")
    parser.add_argument("--run-thresholds", default="0.1,0.2,0.3,0.45,0.6")
    parser.add_argument("--heal-thresholds", default="0.25")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = load_config()
    run_thrs = [float(x) for x in args.run_thresholds.split(",")]
    heal_thrs = [float(x) for x in args.heal_thresholds.split(",")]
    genomes = sweep_genomes(run_thrs, heal_thrs)
    wm = args.worldmap_in if args.worldmap_in and Path(args.worldmap_in).exists() else None

    summary = harvest(
        cfg,
        in_state=str(Path(args.in_state).resolve()),
        state_label=args.state_label,
        genomes=genomes,
        max_steps=args.max_steps,
        worldmap_in=wm,
        out_dir=Path(args.out_dir),
    )
    print(f"[harvest] summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
