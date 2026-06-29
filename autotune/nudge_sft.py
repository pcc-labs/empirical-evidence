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

from autotune.forest_story import ForestVerdict
from autotune.genome import PARAM_DESCRIPTIONS, clamp_params
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


# --------------------------------------------------------------------------- #
# Forest-keyed SFT: same rejection-sampling shape, keyed on the forest reward  #
# --------------------------------------------------------------------------- #
#
# The forest follower fixes navigation (the lever the nav genome can't move) and lets the genome
# vary BATTLE/SURVIVAL — flee-or-fight wild encounters, when to heal, how to pick moves. So forest
# crossings from one start differ by how far the lead survives, and "do more of what passed" means
# emitting the genome that survived furthest. Identical pair logic to ``build_dataset`` above, but
# scored by :class:`autotune.forest_story.ForestVerdict` instead of the map-grained story.

_FOREST_SYSTEM = (
    "You tune a Pokemon Red agent's battle/survival genome to cross Viridian Forest to Pewter "
    "City. Navigation is hand-driven; your genome decides survival. Given the situation, respond "
    "with ONLY the JSON genome that survives furthest through the forest."
)


@dataclass(frozen=True)
class ForestWinner:
    """A forest rollout's genome paired with its forest verdict and fitness (for the turns key)."""

    params: dict
    verdict: ForestVerdict
    fitness: dict


def _forest_rank(w: ForestWinner) -> tuple[float, float, float]:
    """Rank forest crossings by reward, then catchers beaten, then fewer turns.

    Navigation is fixed by the follower, so crossings differ by survival: more sub-beats reached
    (reward), more catchers beaten, or the same reached in fewer turns is the better target. The
    secondary keys keep a gradient even when two genomes tie on the sub-beat count.
    """
    return (
        w.verdict.reward,
        float(w.verdict.signals.trainer_wins),
        -float(w.fitness.get("turns", 1e9)),
    )


def build_forest_mutation_prompt(params: dict, verdict: ForestVerdict) -> str:
    """Ask a model to propose ONE improved survival genome for crossing Viridian Forest.

    The user turn the loop would use at inference; pairs with the flat genome JSON target so there
    is no train/inference skew (mirrors ``build_pair_example`` for the map story).
    """
    desc = "\n".join(f"- {k}: {v}" for k, v in PARAM_DESCRIPTIONS.items())
    return f"""You tune a Pokemon Red agent's genome to cross Viridian Forest to Pewter City.
Navigation is hand-driven along the route; your genome controls battle and survival — whether to
flee or fight wild encounters (hp_run_threshold), when to heal (hp_heal_threshold), and how to
pick moves. Surviving the catchers and the wild grass is what gets the lead to the exit.

This rollout reached forest beat {verdict.furthest_beat} '{verdict.furthest_beat_name}'
(reward {verdict.reward} sub-beats, {verdict.signals.trainer_wins} catchers beaten,
crossed={verdict.crossed}).

Current genome:
{json.dumps(params, indent=2)}

Parameter descriptions:
{desc}

Propose ONE modified genome that survives further through the forest (beat more catchers, avoid
fainting, reach the exit). Return ONLY valid JSON with the same keys, nothing else."""


def build_forest_pair_example(source: ForestWinner, target_params: dict) -> dict:
    """One chat example: improve ``source``'s genome -> the stronger ``target_params``."""
    answer = clamp_params(target_params)
    return {
        "messages": [
            {"role": "system", "content": _FOREST_SYSTEM},
            {
                "role": "user",
                "content": build_forest_mutation_prompt(source.params, source.verdict),
            },
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }


def build_forest_dataset(winners: list[ForestWinner]) -> list[dict]:
    """Rejection-sample forest crossings into improvement pairs (weaker genome -> strongest).

    Keeps only in-forest runs (reward >= 1 == entered), picks the strongest as the target, and
    pairs every strictly-weaker crossing with it. Returns ``[]`` when there is no gradient (fewer
    than two in-forest runs, or all tied at the top). Callers should pass winners that share a
    start state so the pairs are comparable.
    """
    in_forest = [w for w in winners if w.verdict.reward >= 1]
    if len(in_forest) < 2:
        return []
    target = max(in_forest, key=_forest_rank)
    target_rank = _forest_rank(target)
    return [
        build_forest_pair_example(w, target.params)
        for w in in_forest
        if _forest_rank(w) < target_rank and w.params != target.params
    ]


def assemble_forest_corpus(winners_by_state: dict[str, list[ForestWinner]]) -> list[dict]:
    """Build a forest corpus from winners grouped by start state. Pairs form within each state."""
    examples: list[dict] = []
    for state in sorted(winners_by_state):
        examples.extend(build_forest_dataset(winners_by_state[state]))
    return examples
