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
  8. exit to Route 2 / Pewter     (a later map id is visited AND all bug catchers are defeated —
                                   you can't leave the forest without beating them)

The reward is the count of sub-beats reached — a smooth ladder the nudge can climb even on runs
that don't fully cross. Beats are counted whether or not earlier beats fired, so the
trainer/sign/exit gradient survives the item beats: until pokemon-kafka emits ``bag_count`` (an
optional field on ``overworld`` events) the item beats (2, 5, 7) score 0, but the beats past them
still count. ``furthest_beat`` separately reports the in-order frontier. Pure logic: takes
already-loaded telemetry events, no emulator.
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

# You cannot leave Viridian Forest without defeating its bug catchers — a precondition the
# automated discovery engine never surfaced (it got the agent one tile from the exit and stalled).
# Gate the exit beat on it, derived from the catcher beats above so adding a #3 keeps it in sync.
REQUIRED_BUG_CATCHERS = sum(1 for b in FOREST_BEATS if "bug catcher" in b.name.lower())


@dataclass(frozen=True)
class ForestVerdict:
    furthest_beat: int  # in-order frontier reached (0 = not even in the forest); reporting only
    furthest_beat_name: str
    beats_passed: int  # count of beats reached (== sum(per_beat)); the dense reward
    per_beat: tuple[int, ...]  # pass=1/fail=0 for each beat 1..8, marked when REACHED (any order)
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
        8: s.exited and s.trainer_wins >= REQUIRED_BUG_CATCHERS,
    }[beat_id]


def score_forest(events: list[dict]) -> ForestVerdict:
    """Score a rollout's forest crossing as a dense, in-order sub-beat reward. Pure."""
    s = extract_forest_signals(events)
    reached = {b.beat_id for b in FOREST_BEATS if _beat_reached(b.beat_id, s)}

    # Reward = COUNT of beats reached, not the strict in-order frontier. Production telemetry never
    # carries bag_count, so the item beats (2, 5, 7) are permanent gaps; an in-order frontier would
    # cap every in-forest run at 1 and hand the loop a flat reward with no gradient. Counting
    # reached beats credits the catcher/sign/exit progress past those gaps.
    per_beat = tuple(1 if b.beat_id in reached else 0 for b in FOREST_BEATS)
    beats_passed = sum(per_beat)

    # furthest_beat stays the in-order frontier — a faithful "how far, in order" for reporting.
    frontier = 0
    while (frontier + 1) in reached:
        frontier += 1
    name = FOREST_BEATS[frontier - 1].name if frontier >= 1 else "(not in forest)"
    return ForestVerdict(
        furthest_beat=frontier,
        furthest_beat_name=name,
        beats_passed=beats_passed,
        per_beat=per_beat,
        reward=float(beats_passed),
        crossed=s.exited,
        signals=s,
    )
