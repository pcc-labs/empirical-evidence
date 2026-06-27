from autotune.selection import (
    best_verdict,
    is_verdict_better,
    rank_indices,
    select_winner_indices,
)
from tests.conftest import make_verdict


def test_rank_indices_by_reward_then_score():
    verdicts = [
        make_verdict(story_reward=2, score=10),
        make_verdict(story_reward=5, score=1),
        make_verdict(story_reward=5, score=9),
    ]
    assert rank_indices(verdicts) == [2, 1, 0]


def test_select_winners_prefers_on_story():
    verdicts = [
        make_verdict(story_reward=5, on_story=False, score=100),
        make_verdict(story_reward=5, on_story=True, score=1),
    ]
    # Both tie on reward, but the on-story one wins.
    assert select_winner_indices(verdicts) == [1]


def test_select_winners_falls_back_when_none_on_story():
    verdicts = [
        make_verdict(story_reward=3, on_story=False, score=2),
        make_verdict(story_reward=3, on_story=False, score=9),
    ]
    # No on-story winners -> keep all best-reward, ranked by score.
    assert select_winner_indices(verdicts) == [1, 0]


def test_select_winners_empty():
    assert select_winner_indices([]) == []
    assert best_verdict([]) is None


def test_best_verdict_picks_top():
    verdicts = [make_verdict(story_reward=1), make_verdict(story_reward=4)]
    assert best_verdict(verdicts).story_reward == 4


def test_is_verdict_better():
    incumbent = make_verdict(story_reward=4, score=100)
    assert is_verdict_better(make_verdict(story_reward=5, score=1), incumbent) is True
    assert is_verdict_better(make_verdict(story_reward=4, score=200), incumbent) is True
    assert is_verdict_better(make_verdict(story_reward=4, score=50), incumbent) is False
    # None incumbent is always beaten.
    assert is_verdict_better(make_verdict(story_reward=0), None) is True
