"""The story autotune enforces.

A *story* is an ordered sequence of beats the pokemon-kafka agent must hit in order.
It is built from concepts that already exist in pokemon-kafka:

  - the chapter ordering is ``MAP_PROGRESS`` from ``pokemon-kafka/scripts/evolve.py``
    (mirrored here so autotune stays decoupled from pk's ``scripts/`` import path), and
  - per-beat content (display name + waypoints) is read from ``references/routes.json``.

The default story is the canonical Route-1 opening of Pokemon Red (map IDs are
identical across Red/Blue/Yellow, so the same spec scores all three games):
Player's house -> Oak's Lab -> Pallet Town -> Route 1 -> Viridian City -> Route 2 ->
Viridian Forest -> Pewter City.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Mirrors pokemon-kafka/scripts/evolve.py::MAP_PROGRESS — the ordered story chapters.
# map_id -> progress rank (1 = earliest). Keep in sync with evolve.py if pk's path changes.
MAP_PROGRESS: dict[int, int] = {
    37: 1,  # Player's house 1F
    38: 2,  # Player's house 2F
    40: 3,  # Oak's Lab
    0: 4,   # Pallet Town
    12: 5,  # Route 1
    1: 6,   # Viridian City
    13: 7,  # Route 2
    51: 8,  # Viridian Forest
    2: 9,   # Pewter City
}

# Names for maps with no waypoints in routes.json (interiors).
_FALLBACK_NAMES: dict[int, str] = {
    37: "Player's House 1F",
    38: "Player's House 2F",
    40: "Oak's Lab",
    0: "Pallet Town",
    12: "Route 1",
    1: "Viridian City",
    13: "Route 2",
    51: "Viridian Forest",
    2: "Pewter City",
}

# Human-facing milestone text for notable beats (used in nudge prompts / logs).
_MILESTONES: dict[int, str] = {
    40: "Get a starter from Oak",
    1: "Reach Viridian City",
    51: "Cross Viridian Forest",
    2: "Reach Pewter City",
}


@dataclass(frozen=True)
class StoryBeat:
    """One ordered step in the story."""

    beat_id: int  # 1-based ordinal position within the story
    map_id: int
    progress: int  # MAP_PROGRESS rank
    name: str
    waypoints: tuple[dict, ...] = ()
    milestone: str | None = None


@dataclass(frozen=True)
class Story:
    """An ordered list of beats plus the beat that marks completion."""

    name: str
    beats: tuple[StoryBeat, ...]
    target_beat_id: int

    @property
    def target_beat(self) -> StoryBeat:
        return self.beats[self.target_beat_id - 1]

    def beat_for_map(self, map_id: int) -> StoryBeat | None:
        """The beat whose map this is, or None if the map is off-story."""
        for beat in self.beats:
            if beat.map_id == map_id:
                return beat
        return None

    def ordinal_for_map(self, map_id: int) -> int | None:
        """1-based position of a map in the story, or None if off-story."""
        beat = self.beat_for_map(map_id)
        return beat.beat_id if beat else None


def load_routes(routes_json: Path) -> dict[int, dict]:
    """Load ``references/routes.json`` into ``{map_id: {name, waypoints, ...}}``.

    Keys in the file are strings; the ``_comment`` entry is skipped. Missing/unreadable
    files yield an empty mapping so a story can still be built from fallbacks.
    """
    try:
        raw = json.loads(Path(routes_json).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, dict] = {}
    for key, value in raw.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        try:
            out[int(key)] = value
        except ValueError:
            continue
    return out


def build_story(name: str, target_map_id: int, routes: dict[int, dict]) -> Story:
    """Assemble an ordered Story from the chapter ranking and route content."""
    ordered_maps = sorted(MAP_PROGRESS, key=lambda m: MAP_PROGRESS[m])
    beats: list[StoryBeat] = []
    for ordinal, map_id in enumerate(ordered_maps, start=1):
        route = routes.get(map_id, {})
        beat_name = route.get("name") or _FALLBACK_NAMES.get(map_id, f"Map {map_id}")
        waypoints = tuple(route.get("waypoints", ()) or ())
        beats.append(
            StoryBeat(
                beat_id=ordinal,
                map_id=map_id,
                progress=MAP_PROGRESS[map_id],
                name=beat_name,
                waypoints=waypoints,
                milestone=_MILESTONES.get(map_id),
            )
        )

    target_ordinal = next((b.beat_id for b in beats if b.map_id == target_map_id), None)
    if target_ordinal is None:
        raise ValueError(
            f"target_map_id={target_map_id} is not a story beat; "
            f"valid beat maps are {sorted(MAP_PROGRESS)}"
        )
    return Story(name=name, beats=tuple(beats), target_beat_id=target_ordinal)


def load_story(routes_json: Path, name: str = "route1", target_map_id: int = 1) -> Story:
    """Convenience loader: read routes.json and build the default story."""
    return build_story(name=name, target_map_id=target_map_id, routes=load_routes(routes_json))


def derive_story_from_run(
    visited_maps: list[int],
    name: str = "observed",
    routes: dict[int, dict] | None = None,
    target_map_id: int | None = None,
) -> Story:
    """Build a Story from the maps a run *actually* visited, in first-visit order.

    Where ``build_story`` asserts the canonical ``MAP_PROGRESS`` ordering, this derives the
    story from observed telemetry — so the enforced sequence is what the agent did, not a
    hardcoded route (the saturation trap from ``docs/experiment-findings.md``). Off-canonical
    maps (e.g. gym interiors) get progress rank 0 but still become beats. ``target_map_id``
    defaults to the last beat reached.
    """
    routes = routes or {}
    seen: list[int] = []
    for map_id in visited_maps:
        if isinstance(map_id, int) and map_id not in seen:
            seen.append(map_id)
    if not seen:
        raise ValueError("no maps visited; cannot derive a story")

    beats: list[StoryBeat] = []
    for ordinal, map_id in enumerate(seen, start=1):
        route = routes.get(map_id, {})
        beat_name = route.get("name") or _FALLBACK_NAMES.get(map_id, f"Map {map_id}")
        waypoints = tuple(route.get("waypoints", ()) or ())
        beats.append(
            StoryBeat(
                beat_id=ordinal,
                map_id=map_id,
                progress=MAP_PROGRESS.get(map_id, 0),
                name=beat_name,
                waypoints=waypoints,
                milestone=_MILESTONES.get(map_id),
            )
        )

    if target_map_id is None:
        target_ordinal = len(beats)
    else:
        target_ordinal = next((b.beat_id for b in beats if b.map_id == target_map_id), len(beats))
    return Story(name=name, beats=tuple(beats), target_beat_id=target_ordinal)
