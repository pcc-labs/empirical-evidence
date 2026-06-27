"""The orchestrator: Try -> Check -> Reward -> Nudge -> loop back to Try.

One generation:
  1. Try     — run N rollouts of the current genome (env stochasticity gives diverse tries).
  2. Check   — verify each rollout against the story.
  3. Reward  — per-beat pass=1/fail=0; pick the winners (rejection sampling).
  4. Nudge   — reinforce what passed:
                 sft   -> add winners to the SFT buffer, train a LoRA adapter (local MLX),
                 steer -> write a nudge to notes.md.
               Then propose the next genome (via the trained model if available, else heuristic).

Stops early when a rollout reaches the target story beat.

IO/orchestration wrapper — exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotune import nudge_sft, nudge_steer, train_sft
from autotune.config import Config, load_config
from autotune.generate import make_proposer
from autotune.genome import base_genome
from autotune.rollout import run_batch
from autotune.selection import best_verdict, is_verdict_better, select_winner_indices
from autotune.story import Story, load_story
from autotune.verifier import verify


def _resolve_notes_path(cfg: Config) -> Path:
    """Where steer nudges + the genome block are written.

    Defaults to pokemon-kafka's own notes.md so the agent can consume them (L2);
    override with AUTOTUNE_NOTES_PATH (e.g. to keep them inside autotune's out/).
    """
    override = os.environ.get("AUTOTUNE_NOTES_PATH")
    return Path(override) if override else cfg.env.pokemon_kafka_dir / "notes.md"


def _write_best_genome(path: Path, genome: dict, verdict) -> None:
    """Persist the best genome found so far (L1) for applying back to pokemon-kafka."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "genome": genome,
        "story_reward": verdict.story_reward,
        "furthest_beat": verdict.furthest_beat_name,
        "reached_target": verdict.reached_target,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _summarize_generation(gen: int, verdicts: list, story: Story) -> None:
    best = best_verdict(verdicts)
    rewards = [v.story_reward for v in verdicts]
    print(
        f"[gen {gen}] rollouts={len(verdicts)} "
        f"rewards={rewards} best='{best.furthest_beat_name}' "
        f"(beat {best.furthest_beat}/{story.target_beat_id}) score={best.score:.0f}"
    )


def run_loop(cfg: Config) -> dict:
    """Run the full loop. Returns a small summary dict."""
    if cfg.loop.mode == "brock":
        from autotune.brock_loop import run_brock_loop

        return run_brock_loop(
            cfg,
            generations=cfg.loop.generations,
            seed=cfg.loop.seed,
        )

    story = load_story(cfg.env.routes_json, cfg.story.name, cfg.story.target_map_id)
    loop = cfg.loop
    notes_path = _resolve_notes_path(cfg)
    best_genome_path = cfg.storage.out_dir / "best_genome.json"

    genome = base_genome()
    sft_buffer: list[nudge_sft.Winner] = []
    history: list[dict] = []
    reached = False
    best_overall = None
    best_overall_params = genome

    for gen in range(loop.generations):
        # 1. Try — N stochastic rollouts of the current genome.
        work_root = cfg.storage.out_dir / "rollouts" / f"gen-{gen}"
        rollouts = run_batch(
            cfg,
            params_list=[dict(genome) for _ in range(loop.n_rollouts)],
            max_turns=loop.max_turns,
            work_root=work_root,
            concurrency=loop.concurrency,
        )

        # 2 + 3. Check + Reward, then select winners.
        verdicts = [verify(story, r.fitness, r.events) for r in rollouts]
        _summarize_generation(gen, verdicts, story)
        winner_idx = select_winner_indices(verdicts)
        best = best_verdict(verdicts)
        best_params = rollouts[winner_idx[0]].params if winner_idx else genome
        history.append(
            {"generation": gen, "reward": best.story_reward, "furthest": best.furthest_beat_name}
        )

        # L1: track the best genome across all generations and persist it.
        if is_verdict_better(best, best_overall):
            best_overall, best_overall_params = best, best_params
            _write_best_genome(best_genome_path, best_overall_params, best_overall)

        if best.reached_target:
            print(f"[gen {gen}] story enforced — reached target beat '{best.furthest_beat_name}'.")
            reached = True
            break

        winners = [nudge_sft.Winner(rollouts[i].params, verdicts[i]) for i in winner_idx]

        # 4. Nudge.
        proposer = None
        if loop.nudge in ("sft", "both"):
            sft_buffer.extend(winners)
            examples = nudge_sft.build_dataset(sft_buffer, story)
            if examples:
                nudge_sft.write_sft_data(cfg.storage.sft_dir, examples, seed=loop.seed)
                train_sft.train(cfg, cfg.storage.sft_dir, iters=cfg.profile.iters)
                if cfg.storage.adapter_dir.exists():
                    proposer = make_proposer(cfg)
        if loop.nudge in ("steer", "both"):
            line = nudge_steer.write_nudge_note(notes_path, best, best_params)
            # L2: write the machine-readable genome block pokemon-kafka reads at startup.
            nudge_steer.write_genome_block(notes_path, best_params)
            print(f"[gen {gen}] nudge -> {line}")

        # Propose the next genome (model if trained, else heuristic/Claude).
        genome = nudge_steer.propose_next_genome(
            best_params, best, story, proposer=proposer, history=history, seed=gen + loop.seed
        )
        print(f"[gen {gen}] next genome diffs: {nudge_steer.genome_diffs(genome)}")

    return {
        "reached_target": reached,
        "generations_run": len(history),
        "history": history,
        "best_genome": best_overall_params,
        "best_genome_path": str(best_genome_path),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="autotune Try->Check->Reward->Nudge loop.")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="rollouts per generation")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--nudge", choices=["sft", "steer", "both"], default=None)
    parser.add_argument("--mode", choices=["story", "brock"], default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    overrides = {}
    if args.generations is not None:
        overrides["generations"] = args.generations
    if args.n is not None:
        overrides["n_rollouts"] = args.n
    if args.max_turns is not None:
        overrides["max_turns"] = args.max_turns
    if args.nudge is not None:
        overrides["nudge"] = args.nudge
    if args.mode is not None:
        overrides["mode"] = args.mode
    if overrides:
        cfg = cfg.with_loop(**overrides)

    summary = run_loop(cfg)
    if cfg.loop.mode == "brock":
        print(
            f"[loop] done (brock): won={summary['brock_won']} turns={summary['brock_turns']} "
            f"best_level={summary['best_level']} evals={summary['evaluations']}"
        )
        print("[loop] best config -> out/best_brock.json (apply with: scripts/apply_brock.sh)")
        return 0
    print(
        f"[loop] done: {summary['generations_run']} generations, "
        f"reached={summary['reached_target']}"
    )
    print(f"[loop] best genome -> {summary['best_genome_path']}")
    print("[loop] apply it to pokemon-kafka with: scripts/apply_genome.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
