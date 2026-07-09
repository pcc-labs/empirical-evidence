"""Nudge #2: steer the existing param/Claude agent from what passed.

No weight training. We take the winning genome ("do more of what passed") and:
  - propose the next genome to try (exploit on success, perturb when blocked), and
  - append a natural-language nudge to ``notes.md`` for the next attempt.

A ``proposer`` callable ``(prompt: str) -> str`` can be injected to let an LLM (Claude) or the
locally-trained MLX model propose the genome instead of the deterministic heuristic — this is
the shared seam between path 1 (local model) and path 2 (Claude). Prompt construction mirrors
pokemon-kafka/scripts/evolve.py::build_mutation_prompt.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from autotune.config import game_label
from autotune.genome import DEFAULT_PARAMS, PARAM_BOUNDS, PARAM_DESCRIPTIONS, clamp_params
from autotune.story import Story
from autotune.verifier import RolloutVerdict

Proposer = Callable[[str], str]


def genome_diffs(params: dict) -> dict:
    """Params that differ from the default genome (what makes this rollout distinctive)."""
    return {k: v for k, v in params.items() if v != DEFAULT_PARAMS.get(k)}


def build_mutation_prompt(
    params: dict,
    verdict: RolloutVerdict,
    story: Story,
    history: list[dict] | None = None,
) -> str:
    """Ask a model to propose ONE improved genome as JSON. Mirrors evolve.py's prompt."""
    target = story.target_beat
    hist_section = ""
    if history:
        lines = [
            f"  gen {h.get('generation')}: reward={h.get('reward')} furthest='{h.get('furthest')}'"
            for h in history[-10:]
        ]
        hist_section = "\nPrevious generations:\n" + "\n".join(lines) + "\n"

    desc = "\n".join(f"- {k}: {v}" for k, v in PARAM_DESCRIPTIONS.items())
    label = game_label()
    return f"""You are tuning navigation parameters for a Pokemon {label} agent to enforce a story.

Story goal: reach beat {target.beat_id} '{target.name}' (map {target.map_id}), in order.
This rollout reached beat {verdict.furthest_beat} '{verdict.furthest_beat_name}'
(story reward {verdict.story_reward}, on_story={verdict.on_story}).

Current genome:
{json.dumps(params, indent=2)}

Current fitness:
{json.dumps(verdict.fitness, indent=2)}
{hist_section}
Parameter descriptions:
{desc}

Propose ONE modified genome that advances further along the story (reduce stuck_count,
reach later beats in order). Return ONLY valid JSON with the same keys, nothing else."""


def parse_genome_response(text: str | None) -> dict | None:
    """Extract a clamped genome dict from model text. Returns None on failure."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(ln for ln in stripped.splitlines() if not ln.startswith("```")).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    genome = {k: v for k, v in parsed.items() if k in PARAM_BOUNDS}
    return clamp_params(genome) if genome else None


def _heuristic_next(winner_params: dict, verdict: RolloutVerdict, seed: int) -> dict:
    """Deterministic fallback proposer.

    On success (target reached) keep the winning genome — exploit. Otherwise perturb one or
    two numeric params modestly to escape the block, biased toward less stuck behaviour.
    """
    nxt = clamp_params(dict(winner_params))
    if verdict.reached_target:
        return nxt

    rng = random.Random(seed)
    numeric = [k for k, b in PARAM_BOUNDS.items() if not all(isinstance(v, str) for v in b)]
    for key in rng.sample(numeric, k=min(2, len(numeric))):
        lo, hi, typ = PARAM_BOUNDS[key]
        span = hi - lo
        step = rng.choice([-1, 1]) * (span * 0.2)
        nxt[key] = typ(nxt.get(key, DEFAULT_PARAMS[key]) + step)
    return clamp_params(nxt)


def propose_next_genome(
    winner_params: dict,
    verdict: RolloutVerdict,
    story: Story,
    *,
    proposer: Proposer | None = None,
    history: list[dict] | None = None,
    seed: int = 0,
) -> dict:
    """Propose the next genome, via the injected proposer or the heuristic fallback."""
    if proposer is not None:
        prompt = build_mutation_prompt(winner_params, verdict, story, history)
        parsed = parse_genome_response(proposer(prompt))
        if parsed is not None:
            return parsed
    return _heuristic_next(winner_params, verdict, seed)


# Machine-readable genome block embedded in notes.md. pokemon-kafka's agent parses the JSON
# between these markers and uses it as its EVOLVE_PARAMS baseline (env still overrides).
_GENOME_BEGIN = "<!-- autotune:genome"
_GENOME_END = "-->"
_GENOME_BLOCK_RE = re.compile(
    re.escape(_GENOME_BEGIN) + r".*?" + re.escape(_GENOME_END) + r"\n?",
    re.DOTALL,
)


def render_genome_block(genome: dict) -> str:
    """Render the machine-readable genome block written into notes.md."""
    return f"{_GENOME_BEGIN}\n{json.dumps(genome)}\n{_GENOME_END}\n"


def write_genome_block(notes_path: Path, genome: dict) -> None:
    """Write/replace the autotune genome block in notes.md (creating the file if needed)."""
    path = Path(notes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Agent Notes\n"
    stripped = _GENOME_BLOCK_RE.sub("", existing).rstrip("\n")
    path.write_text(stripped + "\n\n" + render_genome_block(genome), encoding="utf-8")


def write_nudge_note(
    notes_path: Path,
    verdict: RolloutVerdict,
    params: dict,
    *,
    stamp: str | None = None,
) -> str:
    """Append a dated nudge to notes.md describing what passed. Returns the line written."""
    when = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    diffs = genome_diffs(params)
    line = (
        f"- [{when}] Reached '{verdict.furthest_beat_name}' "
        f"(beat {verdict.furthest_beat}, reward {verdict.story_reward}). "
        f"Keep genome diffs: {json.dumps(diffs)}"
    )
    path = Path(notes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line
