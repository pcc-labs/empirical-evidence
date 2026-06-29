"""Tests for the Brock loop's pure helpers (state-ladder discovery + leaderboard entry shape).

The full loop (run_brock_loop) drives subprocess rollouts and is exercised by the smoke run.
"""

from __future__ import annotations

from autotune.brock import BrockVerdict
from autotune.brock_loop import Matchup, _entry, discover_states


def _touch(path):
    path.write_bytes(b"")


def test_discover_states_builds_level_ladder_sorted(tmp_path):
    for name in ("lead_lv16.state", "lead_lv10.state", "lead_lv14.state"):
        _touch(tmp_path / name)
    ladder = discover_states(tmp_path)
    assert [m.level for m in ladder] == [10, 14, 16]  # ascending


def test_discover_states_prefers_leveled_over_bare(tmp_path):
    _touch(tmp_path / "lead_lv12.state")
    _touch(tmp_path / "random.state")  # no level -> excluded when leveled exist
    ladder = discover_states(tmp_path)
    assert [m.level for m in ladder] == [12]


def test_discover_states_falls_back_to_single_state(tmp_path):
    fallback = tmp_path / "pre_brock.state"
    _touch(fallback)
    ladder = discover_states(tmp_path / "missing", fallback_state=str(fallback))
    assert len(ladder) == 1
    assert ladder[0].level is None


def test_discover_states_empty(tmp_path):
    assert discover_states(tmp_path / "nope", fallback_state=str(tmp_path / "nope.state")) == []


def test_entry_shape():
    v = BrockVerdict(
        reached_pewter=True, won=True, turns=9, damage_frac=1.0, whiteout=False,
        nav_progress=1.0, lead_species="Squirtle", lead_level=14,
        story_reward=11.82, score=0.9, fitness={},
    )
    genome = {"unknown_move_score": 15.0, "status_move_score": 0.0,
              "hp_run_threshold": 0.2, "hp_heal_threshold": 0.3}
    e = _entry(v, genome, Matchup(14, "/s.state"), "out/x")
    assert e["matchup"] == {"level": 14, "state_path": "/s.state"}
    assert e["won"] is True and e["turns"] == 9
    assert e["reward"] == 11.82
    assert set(e["genome"]) == {"unknown_move_score", "status_move_score",
                                "hp_run_threshold", "hp_heal_threshold"}
