"""Nudge #1 (data side): rejection-sampling SFT examples from rollouts that passed.

"Do more of what passed" = supervised fine-tune the local MLX model to emit a *better* genome
given the situation. The supervised signal is an **improvement pair**: from a weaker genome (and
the story state it reached) the model learns to emit the strongest genome found from the same
start. This mirrors how the loop actually queries the model — ``propose_next_genome`` hands it
the current best via ``nudge_steer.build_mutation_prompt`` and asks for an improved genome — so
the training prompt and the inference prompt are the **same shape** (no train/inference skew), and
the assistant target is a **flat** genome JSON that ``nudge_steer.parse_genome_response`` can read
back (a wrapped ``{"genome": ...}`` object would be filtered to empty and silently rejected).

Building the dataset is pure logic and unit-tested; the actual training subprocess lives in
``train_sft.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from autotune.genome import clamp_params
from autotune.nudge_steer import build_mutation_prompt
from autotune.story import Story
from autotune.verifier import RolloutVerdict

_SYSTEM = (
    "You tune a Pokemon Red agent's navigation genome to advance a fixed story, in order. "
    "Given the current situation, respond with ONLY the JSON genome that advances furthest."
)


@dataclass(frozen=True)
class Winner:
    """A rollout's genome paired with its verdict (one explored point in genome space)."""

    params: dict
    verdict: RolloutVerdict


def _rank(winner: Winner) -> tuple[float, float]:
    """Order winners by story progress, then the composite tiebreaker (fewer turns/stuck).

    Using ``score`` as the secondary key means within-beat improvements (same beat, less stuck,
    fewer turns) still form a learnable gradient when every genome ties on ``story_reward``.
    """
    return (winner.verdict.story_reward, winner.verdict.score)


def build_pair_example(source: Winner, target_params: dict, story: Story) -> dict:
    """One MLX-LM chat example: improve ``source``'s genome -> the better ``target_params``.

    The user turn is the exact prompt the loop uses at inference (``build_mutation_prompt``); the
    assistant turn is the flat target genome JSON (parseable by ``parse_genome_response``).
    """
    answer = clamp_params(target_params)
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": build_mutation_prompt(source.params, source.verdict, story),
            },
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }


def build_dataset(winners: list[Winner], story: Story) -> list[dict]:
    """Rejection-sample ``winners`` into improvement-pair examples.

    Keeps only on-story winners, picks the strongest as the target, and pairs every weaker winner
    with it (``weaker genome -> best genome``). Returns ``[]`` when there is no gradient to teach
    (fewer than two on-story winners, or all tied at the top). Callers should pass winners that
    share a starting point (e.g. one seed state, or one loop's full-run rollouts) so the pairs are
    comparable.
    """
    on_story = [w for w in winners if w.verdict.on_story]
    if len(on_story) < 2:
        return []
    target = max(on_story, key=_rank)
    target_rank = _rank(target)
    return [
        build_pair_example(w, target.params, story)
        for w in on_story
        if _rank(w) < target_rank and w.params != target.params
    ]


def assemble_corpus(winners_by_state: dict[str, list[Winner]], story: Story) -> list[dict]:
    """Build a corpus from winners grouped by seed state.

    Pairs are formed *within* each state (genomes from different starts aren't comparable), then
    concatenated. State order is sorted for determinism.
    """
    examples: list[dict] = []
    for state in sorted(winners_by_state):
        examples.extend(build_dataset(winners_by_state[state], story))
    return examples


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
