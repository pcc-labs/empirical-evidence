"""Unit tests for the pure-logic half of the win-probability harness.

Collection (``collect_rows`` / ``main``) drives the emulator and is smoke-tested, not here.
"""

from autotune.winprob import (
    build_matrix,
    empirical_winrate,
    featurize,
    fit_winprob,
    summarize,
)


def _row(level_gap, won, hp_frac=1.0, enemy_type="bug", had_healing=False):
    """A battle_outcome-shaped row. user_level/enemy_level encode the gap; HP encodes the buffer."""
    return {
        "user_species": "CHARMANDER",
        "user_level": 10 + level_gap,
        "enemy_level": 10,
        "level_gap": level_gap,
        "user_hp_start": int(round(hp_frac * 20)),
        "user_max_hp": 20,
        "had_healing": had_healing,
        "enemy_type": enemy_type,
        "won": won,
    }


# --- featurize ---------------------------------------------------------------


def test_featurize_extracts_features_and_label():
    f = featurize(_row(level_gap=3, won=True, hp_frac=0.5, had_healing=True))
    assert f["level_gap"] == 3.0
    assert f["hp_frac"] == 0.5
    assert f["had_healing"] == 1.0
    assert f["enemy_type"] == "bug"
    assert f["won"] == 1


def test_featurize_clamps_hp_frac_and_guards_zero_maxhp():
    # hp_start > max_hp clamps to 1.0; max_hp 0 must not divide-by-zero.
    assert featurize({"user_hp_start": 30, "user_max_hp": 20})["hp_frac"] == 1.0
    assert featurize({"user_hp_start": 5, "user_max_hp": 0})["hp_frac"] == 1.0


# --- build_matrix ------------------------------------------------------------


def test_build_matrix_shape_and_enemy_onehot():
    rows = [_row(1, True, enemy_type="bug"), _row(-2, False, enemy_type="normal")]
    X, y, columns = build_matrix(rows)
    assert columns == ["level_gap", "hp_frac", "had_healing", "enemy:bug", "enemy:normal"]
    assert X.shape == (2, 5)
    # row0 is bug -> the bug one-hot is set, normal is not.
    assert X[0, 3] == 1.0 and X[0, 4] == 0.0
    assert X[1, 3] == 0.0 and X[1, 4] == 1.0
    assert list(y) == [1.0, 0.0]


# --- fit_winprob -------------------------------------------------------------


def test_fit_learns_level_gap_separation():
    # Cleanly separable: win iff level_gap >= 0. The model should put P(win) high above the
    # threshold and low below it.
    rows = []
    for gap in range(-6, 7):
        for _ in range(8):
            rows.append(_row(level_gap=gap, won=(gap >= 0)))
    model = fit_winprob(rows)
    assert model.predict_proba(_row(level_gap=5, won=False)) > 0.8
    assert model.predict_proba(_row(level_gap=-5, won=True)) < 0.2


def test_fit_learns_matchup_from_enemy_type():
    # Same level gap, but one enemy type always loses to us and another always beats us. With
    # enemy_type as a raw one-hot, the model discovers the matchup from outcomes alone.
    rows = []
    for _ in range(40):
        rows.append(_row(level_gap=0, won=True, enemy_type="bug"))      # we always win vs bug
        rows.append(_row(level_gap=0, won=False, enemy_type="water"))   # we always lose vs water
    model = fit_winprob(rows)
    assert model.predict_proba(_row(0, won=False, enemy_type="bug")) > 0.7
    assert model.predict_proba(_row(0, won=True, enemy_type="water")) < 0.3


def test_fit_raises_on_empty():
    import pytest

    with pytest.raises(ValueError):
        fit_winprob([])


# --- empirical_winrate / summarize ------------------------------------------


def test_empirical_winrate_buckets_by_level_gap():
    rows = [_row(2, True), _row(2, True), _row(2, False), _row(-1, False)]
    table = empirical_winrate(rows, "level_gap")
    assert table[2] == {"n": 3, "wins": 2, "win_rate": round(2 / 3, 3)}
    assert table[-1] == {"n": 1, "wins": 0, "win_rate": 0.0}


def test_summarize_counts_and_curve():
    rows = [_row(1, True), _row(1, False), _row(-3, False, enemy_type="normal")]
    s = summarize(rows)
    assert s["rows"] == 3 and s["wins"] == 1 and s["losses"] == 2
    assert s["win_rate"] == round(1 / 3, 3)
    assert set(s["enemy_types"]) == {"bug", "normal"}
    assert s["by_level_gap"][1]["n"] == 2
