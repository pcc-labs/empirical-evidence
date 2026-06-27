"""Nudge #1 (data side): rejection-sampling SFT examples from rollouts that passed.

"Do more of what passed" = supervised fine-tune the local MLX model to emit the genome that
the winning rollout used, given the story situation. Each winning rollout becomes one chat
example (situation -> winning genome). The examples are written in MLX-LM's chat format
(``{"messages": [...]}``) so ``train_sft.py`` can train a LoRA adapter on them.

Building the dataset is pure logic and unit-tested; the actual training subprocess lives in
``train_sft.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from autotune.story import Story
from autotune.verifier import RolloutVerdict

_SYSTEM = (
    "You tune a Pokemon Red agent's navigation genome to advance a fixed story, in order. "
    "Given the current situation, respond with ONLY the JSON genome that advances furthest."
)

# Fitness fields worth showing the model as the "situation".
_FITNESS_KEYS = ("final_map_id", "maps_visited", "badges", "party_size", "stuck_count", "turns")


@dataclass(frozen=True)
class Winner:
    """A winning rollout's genome paired with its verdict."""

    params: dict
    verdict: RolloutVerdict


def summarize_fitness(fitness: dict) -> dict:
    """Compact, model-facing subset of a fitness dict."""
    return {k: fitness.get(k) for k in _FITNESS_KEYS if k in fitness}


def build_situation(verdict: RolloutVerdict, story: Story) -> str:
    """The user-message description of where the run was and where the story is going."""
    target = story.target_beat
    return (
        f"Story target: beat {target.beat_id} '{target.name}' (map {target.map_id}), in order.\n"
        f"Reached: beat {verdict.furthest_beat} '{verdict.furthest_beat_name}' "
        f"(reward {verdict.story_reward}, on_story={verdict.on_story}).\n"
        f"Fitness: {json.dumps(summarize_fitness(verdict.fitness))}\n"
        f"Propose the genome."
    )


def build_example(winner: Winner, story: Story) -> dict:
    """One MLX-LM chat example: situation -> winning genome (with a one-line rationale)."""
    answer = {
        "genome": winner.params,
        "rationale": (
            f"Reached '{winner.verdict.furthest_beat_name}' "
            f"with story reward {winner.verdict.story_reward}."
        ),
    }
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": build_situation(winner.verdict, story)},
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }


def build_dataset(winners: list[Winner], story: Story) -> list[dict]:
    """Build chat examples from winners, skipping off-story ones."""
    return [build_example(w, story) for w in winners if w.verdict.on_story]


def split_train_valid(
    examples: list[dict], valid_frac: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/valid split. Always leaves >=1 example in train if any exist."""
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    if len(rows) < 2:
        return rows, []
    n_valid = max(1, int(len(rows) * valid_frac))
    n_valid = min(n_valid, len(rows) - 1)
    return rows[n_valid:], rows[:n_valid]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sft_data(
    sft_dir: Path,
    examples: list[dict],
    valid_frac: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write ``train.jsonl`` and ``valid.jsonl`` under ``sft_dir`` (MLX-LM data dir)."""
    train, valid = split_train_valid(examples, valid_frac, seed)
    sft_dir = Path(sft_dir)
    train_path = sft_dir / "train.jsonl"
    valid_path = sft_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    # MLX-LM still expects valid.jsonl to exist; mirror train when too few examples to split.
    write_jsonl(valid_path, valid or train)
    return train_path, valid_path
