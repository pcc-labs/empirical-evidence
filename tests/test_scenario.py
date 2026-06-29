import random

from autotune.genome import DEFAULT_PARAMS, PARAM_BOUNDS
from autotune.scenario import (
    BATTLE_PARAM_KEYS,
    NAV_PARAM_KEYS,
    SCENARIOS,
    bad_battle_genome,
    bad_nav_genome,
    battle_reward,
    mean_reward,
    mutate_battle_params,
    mutate_nav_params,
    nav_reward,
)


def test_bad_battle_genome_is_bad_but_valid():
    g = bad_battle_genome()
    assert g["status_move_score"] == 10.0
    assert g["hp_run_threshold"] == 0.5
    for k in BATTLE_PARAM_KEYS:
        lo, hi, _ = PARAM_BOUNDS[k]
        assert lo <= g[k] <= hi
    # Navigation params stay at defaults so the agent can still reach/handle the battle.
    assert g["stuck_threshold"] == DEFAULT_PARAMS["stuck_threshold"]


def test_battle_reward():
    assert battle_reward({"battles_won": 1}) == 1.0
    assert battle_reward({"battles_won": 3}) == 1.0
    assert battle_reward({"battles_won": 0}) == 0.0
    assert battle_reward({}) == 0.0


def test_mean_reward():
    assert mean_reward([{"battles_won": 1}, {"battles_won": 0}], battle_reward) == 0.5
    assert mean_reward([{"battles_won": 1}, {"battles_won": 1}], battle_reward) == 1.0
    assert mean_reward([], battle_reward) == 0.0
    assert mean_reward([{"maps_visited": 1}, {"maps_visited": 3}], nav_reward) == 2.0


def test_mutate_count_and_bounds():
    g = bad_battle_genome()
    cands = mutate_battle_params(g, random.Random(1), 5)
    assert len(cands) == 5
    for c in cands:
        for k in BATTLE_PARAM_KEYS:
            lo, hi, _ = PARAM_BOUNDS[k]
            assert lo <= c[k] <= hi


def test_mutate_only_touches_battle_params():
    g = bad_battle_genome()
    nav_keys = [k for k in g if k not in BATTLE_PARAM_KEYS]
    for c in mutate_battle_params(g, random.Random(2), 4):
        for k in nav_keys:
            assert c[k] == g[k]


def test_mutate_deterministic():
    g = bad_battle_genome()
    a = mutate_battle_params(g, random.Random(7), 3)
    b = mutate_battle_params(g, random.Random(7), 3)
    assert a == b


# --- navigation scenario ---


def test_bad_nav_genome_is_bad_but_valid():
    g = bad_nav_genome()
    assert g["stuck_threshold"] == 20
    assert g["bt_max_attempts"] == 1
    for k in NAV_PARAM_KEYS:
        lo, hi, _ = PARAM_BOUNDS[k]
        assert lo <= g[k] <= hi


def test_nav_reward_is_maps_visited():
    assert nav_reward({"maps_visited": 3}) == 3.0
    assert nav_reward({}) == 0.0


def test_mutate_nav_only_touches_nav_params():
    g = bad_nav_genome()
    non_nav = [k for k in g if k not in NAV_PARAM_KEYS]
    for c in mutate_nav_params(g, random.Random(3), 4):
        for k in non_nav:
            assert c[k] == g[k]


def test_scenarios_registry():
    assert set(SCENARIOS) == {"battle", "nav", "brock"}
    nav = SCENARIOS["nav"]
    assert nav.battle_limit == 0
    assert nav.reward({"maps_visited": 2}) == 2.0
    assert SCENARIOS["battle"].battle_limit == 1
    # Brock: a fast win scores in the [11, 12) win band; a loss is the coarse reached floor.
    brock = SCENARIOS["brock"]
    assert brock.battle_limit == 1
    assert brock.reward({"brock_won": True, "brock_turns": 10}) > 11.0
    assert brock.reward({"brock_won": False, "brock_turns": 30}) == 1.0
