"""Rejection-sampling selection: "do more of what passed".

Given a group of rollout verdicts, pick the winners the Nudge step should reinforce. The
primary key is the story reward (in-order beats reached); ties break on the composite score.
Pure functions so both Nudge backends and the loop share identical selection semantics.
"""

from __future__ import annotations

from autotune.verifier import RolloutVerdict


def rank_indices(verdicts: list[RolloutVerdict]) -> list[int]:
    """Indices of verdicts, best first (story_reward desc, then score desc)."""
    return sorted(
        range(len(verdicts)),
        key=lambda i: (verdicts[i].story_reward, verdicts[i].score),
        reverse=True,
    )


def select_winner_indices(verdicts: list[RolloutVerdict]) -> list[int]:
    """Indices of the winning rollouts to reinforce.

    Keeps every rollout that tied for the best story reward, preferring on-story ones:
    if any best-reward rollout is on-story, only on-story ones win; otherwise all
    best-reward rollouts win. Empty input -> empty list.
    """
    if not verdicts:
        return []
    best = max(v.story_reward for v in verdicts)
    best_idx = [i for i, v in enumerate(verdicts) if v.story_reward == best]
    on_story = [i for i in best_idx if verdicts[i].on_story]
    winners = on_story or best_idx
    return sorted(winners, key=lambda i: verdicts[i].score, reverse=True)


def best_verdict(verdicts: list[RolloutVerdict]) -> RolloutVerdict | None:
    """The single best verdict in the group, or None if empty."""
    order = rank_indices(verdicts)
    return verdicts[order[0]] if order else None


def is_verdict_better(candidate: RolloutVerdict, incumbent: RolloutVerdict | None) -> bool:
    """True if ``candidate`` beats ``incumbent`` (higher story_reward, then score).

    A ``None`` incumbent is always beaten — used to track the best genome across generations.
    """
    if incumbent is None:
        return True
    return (candidate.story_reward, candidate.score) > (incumbent.story_reward, incumbent.score)
