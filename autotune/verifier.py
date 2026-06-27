"""Check + Reward: turn one rollout's telemetry into a story-shaped, verifiable reward.

This is the slide's "Check (verifier runs the tests) -> Reward (pass=1, fail=0)" made
story-shaped. Instead of pokemon-kafka's single scalar fitness, we score the rollout against
the ordered :mod:`autotune.story`:

  - which story beats were reached, **in order**, and
  - a per-beat ``pass=1 / fail=0`` vector up to the target beat.

``score()`` mirrors pokemon-kafka/scripts/evolve.py::score and is kept only as a tiebreaker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autotune.story import Story

# Mirrors pokemon-kafka/scripts/evolve.py::score — used only to break ties between
# rollouts that reached the same story beat.
_MAP_PROGRESS_WEIGHT = 1000


def composite_score(fitness: dict, story: Story) -> float:
    """pokemon-kafka's composite fitness (progress-weighted), for tiebreaking."""
    map_id = fitness.get("final_map_id", 0)
    progress = next((b.progress for b in story.beats if b.map_id == map_id), 0)
    return (
        progress * _MAP_PROGRESS_WEIGHT
        + fitness.get("badges", 0) * 5000
        + fitness.get("party_size", 0) * 500
        + fitness.get("battles_won", 0) * 100
        - fitness.get("stuck_count", 0) * 5
        - fitness.get("turns", 0) * 0.1
        - fitness.get("backtrack_restores", 0) * 2
    )


@dataclass(frozen=True)
class RolloutVerdict:
    """The verifier's judgement of one rollout."""

    furthest_beat: int  # ordinal of the furthest beat reached IN ORDER (0 = none on-story)
    furthest_beat_name: str
    beats_passed: int  # count of in-order beats up to the target beat (the reward)
    per_beat: tuple[int, ...]  # pass=1/fail=0 for beats 1..target_beat_id
    on_story: bool  # no out-of-order jump past a skipped beat
    reached_target: bool
    story_reward: float  # primary reward = beats_passed
    score: float  # composite tiebreaker
    visited_maps: tuple[int, ...]
    fitness: dict


def extract_visited_maps(events: list[dict]) -> list[int]:
    """Ordered sequence of map ids visited, derived from game telemetry events.

    Uses ``overworld`` (``data.map_id``) and ``map_change`` (``data.prev_map`` then
    ``data.new_map``) events in turn order, collapsing consecutive duplicates.
    """
    ordered = sorted(events, key=lambda e: e.get("turn", 0))
    seq: list[int] = []

    def push(map_id) -> None:
        if isinstance(map_id, int) and (not seq or seq[-1] != map_id):
            seq.append(map_id)

    for ev in ordered:
        data = ev.get("data", {})
        etype = ev.get("event_type")
        if etype == "overworld":
            push(data.get("map_id"))
        elif etype == "map_change":
            push(data.get("prev_map"))
            push(data.get("new_map"))
    return seq


def _furthest_in_order(story: Story, visited_ordinals: list[int]) -> tuple[int, int]:
    """Return (furthest_in_order_ordinal, max_ordinal_seen).

    The in-order frontier is the longest run of consecutive story ordinals, starting from
    the smallest reached ordinal (the agent's start), with no gaps. A later beat reached
    while an intermediate beat was skipped shows up as ``max_ordinal_seen > frontier``.
    """
    if not visited_ordinals:
        return 0, 0
    reached = set(visited_ordinals)
    start = min(reached)
    frontier = start
    while (frontier + 1) in reached:
        frontier += 1
    return frontier, max(reached)


def verify(story: Story, fitness: dict, events: list[dict]) -> RolloutVerdict:
    """Score a rollout against the story. Pure: takes already-loaded fitness + events."""
    visited = extract_visited_maps(events)

    # Fallback when no game telemetry is present: trust fitness.final_map_id as the
    # furthest point, assuming an in-order arrival (degraded but usable).
    if not visited:
        final_map = fitness.get("final_map_id")
        visited = [final_map] if isinstance(final_map, int) else []

    visited_ordinals = [o for o in (story.ordinal_for_map(m) for m in visited) if o is not None]
    frontier, max_seen = _furthest_in_order(story, visited_ordinals)

    target = story.target_beat_id
    per_beat = tuple(1 if b <= frontier else 0 for b in range(1, target + 1))
    beats_passed = sum(per_beat)
    on_story = max_seen == frontier
    reached_target = frontier >= target

    furthest_name = story.beats[frontier - 1].name if frontier >= 1 else "(start)"

    return RolloutVerdict(
        furthest_beat=frontier,
        furthest_beat_name=furthest_name,
        beats_passed=beats_passed,
        per_beat=per_beat,
        on_story=on_story,
        reached_target=reached_target,
        story_reward=float(beats_passed),
        score=composite_score(fitness, story),
        visited_maps=tuple(visited),
        fitness=fitness,
    )


# ---------------------------------------------------------------------------
# Thin IO loaders (used by rollout.py / loop.py; not part of the pure core)
# ---------------------------------------------------------------------------


def load_fitness(path: Path) -> dict:
    """Read a rollout's fitness JSON; empty dict if missing/unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_game_events(telemetry_dir: Path) -> list[dict]:
    """Read all game events from ``<telemetry_dir>/game/*.jsonl``."""
    game_dir = Path(telemetry_dir) / "game"
    events: list[dict] = []
    if not game_dir.is_dir():
        return events
    for path in sorted(game_dir.glob("*.jsonl")):
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
    return events
