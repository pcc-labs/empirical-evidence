"""Dense, observation-based reward for crossing Viridian Forest.

The map-grained :mod:`autotune.story` treats the whole forest (map 51) as ONE beat, so a training
loop gets no gradient until the agent fully exits — the saturation trap that has stalled every
forest attempt. This module breaks the forest into ORDERED sub-beats taken from the human
walkthrough, each triggered by an OBSERVATION in telemetry, never a coordinate:

  1. enter the forest
  2. pick up the Poke Ball        (bag count >= 1)
  3. defeat bug catcher #1        (trainer wins >= 1)
  4. defeat bug catcher #2        (trainer wins >= 2)
  5. pick up the Antidote         (bag count >= 2)
  6. read the Trainer Tips sign   (a discovery event whose text mentions TRAINER/TIPS)
  7. pick up the hidden Potion    (bag count >= 3)
  8. exit to Route 2 / Pewter     (a later map id is visited)

The reward is the count of sub-beats reached IN ORDER — a smooth ladder the nudge can climb even
on runs that don't fully cross. Item beats read an optional ``bag_count`` field on ``overworld``
events; until pokemon-kafka emits it they simply score 0, leaving the trainer/sign/exit gradient
intact. Pure logic: takes already-loaded telemetry events, no emulator.
"""

from __future__ import annotations

from dataclasses import dataclass

FOREST_MAP = 51
EXIT_MAPS = (13, 2)  # Route 2, Pewter City — anything past the forest


@dataclass(frozen=True)
class ForestSignals:
    """Observations distilled from one rollout's telemetry."""

    entered_forest: bool
    trainer_wins: int
    max_bag_count: int
    sign_read: bool
    exited: bool


@dataclass(frozen=True)
class ForestBeat:
    beat_id: int
    name: str


FOREST_BEATS: tuple[ForestBeat, ...] = (
    ForestBeat(1, "Enter Viridian Forest"),
    ForestBeat(2, "Pick up the Poke Ball"),
    ForestBeat(3, "Defeat bug catcher #1"),
    ForestBeat(4, "Defeat bug catcher #2"),
    ForestBeat(5, "Pick up the Antidote"),
    ForestBeat(6, "Read the Trainer Tips sign"),
    ForestBeat(7, "Pick up the hidden Potion"),
    ForestBeat(8, "Exit to Route 2 / Pewter"),
)


@dataclass(frozen=True)
class ForestVerdict:
    furthest_beat: int  # furthest sub-beat reached IN ORDER (0 = not even in the forest)
    furthest_beat_name: str
    beats_passed: int  # == furthest_beat; the in-order reward
    per_beat: tuple[int, ...]  # pass=1/fail=0 for beats 1..8
    reward: float  # = beats_passed (dense ladder)
    crossed: bool
    signals: ForestSignals


_SIGN_KEYWORDS = ("TRAINER", "TIPS")


def extract_forest_signals(events: list[dict]) -> ForestSignals:
    """Distill the forest sub-beat observations from telemetry events. Pure."""
    entered = exited = sign = False
    trainer_wins = 0
    max_bag = 0
    for ev in events:
        etype = ev.get("event_type")
        data = ev.get("data", {}) or {}
        if etype == "overworld":
            if data.get("map_id") == FOREST_MAP:
                entered = True
            if data.get("map_id") in EXIT_MAPS:
                exited = True
            bag = data.get("bag_count")
            if isinstance(bag, int) and bag > max_bag:
                max_bag = bag
        elif etype == "map_change":
            if data.get("new_map") == FOREST_MAP or data.get("prev_map") == FOREST_MAP:
                entered = True
            if data.get("new_map") in EXIT_MAPS:
                exited = True
        elif etype == "battle_outcome":
            if data.get("battle_type") == 2 and data.get("won"):
                trainer_wins += 1
        elif etype == "discovery":
            text = str(data.get("text", "")).upper()
            if any(k in text for k in _SIGN_KEYWORDS):
                sign = True
    return ForestSignals(
        entered_forest=entered,
        trainer_wins=trainer_wins,
        max_bag_count=max_bag,
        sign_read=sign,
        exited=exited,
    )


def _beat_reached(beat_id: int, s: ForestSignals) -> bool:
    return {
        1: s.entered_forest,
        2: s.max_bag_count >= 1,
        3: s.trainer_wins >= 1,
        4: s.trainer_wins >= 2,
        5: s.max_bag_count >= 2,
        6: s.sign_read,
        7: s.max_bag_count >= 3,
        8: s.exited,
    }[beat_id]


def score_forest(events: list[dict]) -> ForestVerdict:
    """Score a rollout's forest crossing as a dense, in-order sub-beat reward. Pure."""
    s = extract_forest_signals(events)
    reached = {b.beat_id for b in FOREST_BEATS if _beat_reached(b.beat_id, s)}
    # in-order frontier: longest run 1,2,3,... with no gap
    frontier = 0
    while (frontier + 1) in reached:
        frontier += 1
    per_beat = tuple(1 if b.beat_id <= frontier else 0 for b in FOREST_BEATS)
    name = FOREST_BEATS[frontier - 1].name if frontier >= 1 else "(not in forest)"
    return ForestVerdict(
        furthest_beat=frontier,
        furthest_beat_name=name,
        beats_passed=frontier,
        per_beat=per_beat,
        reward=float(frontier),
        crossed=s.exited,
        signals=s,
    )
