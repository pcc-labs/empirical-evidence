"""Check + Reward for the Brock fight: turn one rollout's telemetry into a banded,
directional, non-saturated reward — "fewest turns to beat Brock".

``docs/experiment-findings.md`` documents why the route1 reward could not teach anything: it
saturated flat (every rollout scored the same), so rejection sampling had nothing to select on.
This reward avoids that by being *dense in every band*:

  - **didn't reach Pewter**  -> directional ``nav_progress`` in ``[0, 0.9]`` (NOT ``maps_visited``,
    which the findings proved rewards backward wandering),
  - **reached but lost**     -> ``1.0 + damage_frac*2`` in ``[1.0, 3.0)`` (so even an all-loss
    generation has a selectable gradient — how much of Brock's team it chipped),
  - **won**                  -> ``11 + (T_MAX - turns)/T_MAX`` in ``[11, 12)`` (strictly faster
    is strictly better).

The bands are disjoint by construction (``W_DMG < W_WIN``), so the reward is strictly monotone:
not-reached < lost < won, and within each band the dense term breaks ties. ``BrockVerdict``
duck-types the three fields :mod:`autotune.selection` reads (``story_reward``, ``score``,
``on_story``) so the existing winner-selection is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from autotune.story import Story
from autotune.verifier import extract_visited_maps, verify

# Reward band weights. Invariant: W_DMG < W_WIN keeps the lost/won bands disjoint.
W_REACH = 1.0
W_DMG = 2.0
W_WIN = 10.0
# A Brock-fight turn budget (the fight is ~10-40 in-battle turns), NOT the rollout cap —
# using the rollout cap would crush every real fight into a sliver near the band floor.
BROCK_MAX_TURNS = 50

# Pewter City overworld map id (the gym interior is a different map; see story.py).
PEWTER_MAP_ID = 2
# Brock's team size, for normalizing damage dealt into [0, 1].
BROCK_TEAM_SIZE = 2
# Gym-leader-level threshold that distinguishes Brock (Geodude L12 / Onix L14) from the
# low-level Viridian Forest bug-catcher trainers, when the gym map id isn't pinned.
BROCK_LEVEL_FLOOR = 12


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _is_brock_end(data: dict, brock_map_id: int | None) -> bool:
    """Is this ``battle_end`` event the Brock fight? Trainer battle (type 2), identified by
    the gym map id when known, else by a gym-leader-level opponent."""
    if data.get("battle_type") != 2:
        return False
    if brock_map_id is not None:
        return data.get("map_id") == brock_map_id
    return (data.get("opponent_level") or 0) >= BROCK_LEVEL_FLOOR


def _find_brock_end(events: list[dict], brock_map_id: int | None) -> dict | None:
    for ev in events:
        if ev.get("event_type") == "battle_end" and _is_brock_end(ev.get("data", {}), brock_map_id):
            return ev
    return None


def _brock_damage_frac(events: list[dict], brock_end: dict | None) -> float:
    """Fraction of Brock's team chipped away — a dense gradient for the lost band.

    Battle events carry per-turn ``enemy_hp``/``enemy_max_hp``. Each of Brock's Pokemon has a
    distinct ``enemy_max_hp``; segment the fight's battle events by it and, per segment, take the
    lowest HP fraction reached (0.0 = that mon fainted). ``damage = sum(1 - min_frac) / team``.
    """
    if brock_end is None:
        return 0.0
    end_turn = brock_end.get("turn", 0)
    battle_turns = brock_end.get("data", {}).get("battle_turns") or 0
    start_turn = end_turn - battle_turns

    min_frac_by_mon: dict[int, float] = {}
    for ev in events:
        if ev.get("event_type") != "battle":
            continue
        data = ev.get("data", {})
        if data.get("battle_type") != 2:
            continue
        if not (start_turn <= ev.get("turn", 0) <= end_turn):
            continue
        max_hp = data.get("enemy_max_hp") or 0
        if max_hp <= 0:
            continue
        frac = _clamp((data.get("enemy_hp") or 0) / max_hp, 0.0, 1.0)
        min_frac_by_mon[max_hp] = min(min_frac_by_mon.get(max_hp, 1.0), frac)

    if not min_frac_by_mon:
        return 0.0
    dealt = sum(1.0 - f for f in min_frac_by_mon.values())
    return _clamp(dealt / BROCK_TEAM_SIZE, 0.0, 1.0)


def _brock_whiteout(brock_end: dict | None, won: bool) -> bool:
    """Did the agent white out (lost with no party left)? Diagnostic only — kept out of the
    scalar reward so it can never cross a band boundary."""
    if won or brock_end is None:
        return False
    party = brock_end.get("data", {}).get("party") or []
    if party:
        return all((p.get("hp") or 0) == 0 for p in party)
    return True  # lost with no party info recorded — treat as a white-out


def extract_brock_fitness(
    events: list[dict],
    fitness: dict,
    story: Story,
    brock_map_id: int | None = None,
) -> dict:
    """Derive the Brock metrics from a rollout's telemetry + fitness.

    Prefers the agent's authoritative ``brock_*`` fitness fields, falling back to the
    ``battle_end`` telemetry, and derives ``damage_frac``/``whiteout``/``nav_progress``
    (which the agent does not emit) from the event stream.
    """
    visited = extract_visited_maps(events)
    brock_end = _find_brock_end(events, brock_map_id)

    reached_pewter = (
        PEWTER_MAP_ID in visited
        or brock_end is not None
        or fitness.get("final_map_id") == PEWTER_MAP_ID
    )

    won = fitness.get("brock_won")
    if won is None and brock_end is not None:
        won = brock_end.get("data", {}).get("won")
    won = bool(won)

    turns = fitness.get("brock_turns")
    if turns is None and brock_end is not None:
        turns = brock_end.get("data", {}).get("battle_turns")

    # nav_progress reuses the verifier's directional in-order frontier (NOT maps_visited).
    nav = verify(story, fitness, events)
    nav_progress = (
        nav.beats_passed / story.target_beat_id if story.target_beat_id else 0.0
    )

    damage_frac = 1.0 if won else _brock_damage_frac(events, brock_end)
    whiteout = _brock_whiteout(brock_end, won)

    lead_species = fitness.get("brock_lead_species")
    lead_level = fitness.get("brock_lead_level")
    if lead_species is None and brock_end is not None:
        party = brock_end.get("data", {}).get("party") or []
        if party:
            lead_species = party[0].get("species")
            lead_level = party[0].get("level")

    return {
        "reached_pewter": bool(reached_pewter),
        "won": won,
        "turns": turns,
        "damage_frac": float(damage_frac),
        "whiteout": bool(whiteout),
        "nav_progress": float(_clamp(nav_progress, 0.0, 1.0)),
        "lead_species": lead_species,
        "lead_level": lead_level,
    }


def brock_reward(metrics: dict) -> float:
    """Banded, strictly-monotone reward (not-reached < lost < won; faster win > slower win)."""
    if not metrics["reached_pewter"]:
        return _clamp(metrics["nav_progress"], 0.0, 1.0) * (W_REACH * 0.9)  # [0, 0.9]
    if not metrics["won"]:
        return W_REACH + _clamp(metrics["damage_frac"], 0.0, 1.0) * W_DMG  # [1.0, 3.0]
    turns = metrics["turns"] if metrics["turns"] else BROCK_MAX_TURNS
    turns = _clamp(float(turns), 1.0, BROCK_MAX_TURNS)
    speed = (BROCK_MAX_TURNS - turns) / BROCK_MAX_TURNS  # [0, 1)
    return W_REACH + W_WIN + speed  # [11, 12)


def brock_fitness_reward(fitness: dict) -> float:
    """Coarse fitness-only Brock reward for the ``experiment.py`` grid/hill-climb driver.

    The in-loop path uses :func:`verify_brock` (events-aware) for the full banded reward,
    including the damage-fraction loss gradient and nav progress. This proxy only sees the
    agent's ``brock_won``/``brock_turns`` fitness fields, so it is flat across losses — fine
    for ad-hoc discovery from a loaded pre-Brock state, where the win/turns signal dominates.
    """
    if fitness.get("brock_won"):
        turns = fitness.get("brock_turns") or BROCK_MAX_TURNS
        t = _clamp(float(turns), 1.0, BROCK_MAX_TURNS)
        return W_REACH + W_WIN + (BROCK_MAX_TURNS - t) / BROCK_MAX_TURNS
    reached = (
        fitness.get("brock_turns") is not None or fitness.get("final_map_id") == PEWTER_MAP_ID
    )
    return W_REACH if reached else 0.0


def brock_score(metrics: dict) -> float:
    """Float tiebreaker: more damage and fewer turns rank higher; white-outs sink."""
    turns = float(metrics["turns"]) if metrics["turns"] else float(BROCK_MAX_TURNS)
    return metrics["damage_frac"] - 0.01 * turns - (1.0 if metrics["whiteout"] else 0.0)


@dataclass(frozen=True)
class BrockVerdict:
    """The verifier's judgement of one Brock rollout. Duck-types the three fields
    :mod:`autotune.selection` reads (``story_reward``, ``score``, ``on_story``)."""

    reached_pewter: bool
    won: bool
    turns: int | None
    damage_frac: float
    whiteout: bool
    nav_progress: float
    lead_species: str | None
    lead_level: int | None
    story_reward: float
    score: float
    fitness: dict

    @property
    def on_story(self) -> bool:
        return self.reached_pewter


def verify_brock(
    story: Story,
    fitness: dict,
    events: list[dict],
    brock_map_id: int | None = None,
) -> BrockVerdict:
    """Score a Brock rollout. Pure: takes already-loaded fitness + events."""
    m = extract_brock_fitness(events, fitness, story, brock_map_id)
    return BrockVerdict(
        story_reward=brock_reward(m),
        score=brock_score(m),
        fitness=fitness,
        **m,
    )
