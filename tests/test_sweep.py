"""Unit tests for the pure parts of the prep sweep: config design + ranking.

``run_sweep`` (the subprocess driver) is exercised by the real run, not here — same convention as
``rollout.py`` / ``speedrun.py``.
"""

from __future__ import annotations

import pytest

from autotune.genome import DEFAULT_PARAMS, PARAM_BOUNDS, base_genome
from autotune.sweep import build_report, build_sweep, rank_sweep


def _within_bounds(genome: dict) -> bool:
    for key, value in genome.items():
        bounds = PARAM_BOUNDS.get(key)
        if bounds is None:
            continue
        if all(isinstance(v, str) for v in bounds):  # enum
            if value not in bounds:
                return False
            continue
        lo, hi, _typ = bounds
        if not (lo <= value <= hi):
            return False
    return True


def test_build_sweep_count_and_unique_labels():
    configs = build_sweep(10)
    assert len(configs) == 10
    labels = [c.label for c in configs]
    assert len(set(labels)) == 10


def test_control_is_the_default_genome():
    configs = build_sweep(10)
    control = configs[0]
    assert control.label == "control"
    assert control.strategy == "medium"
    assert control.genome == base_genome()
    assert control.diffs() == {}


def test_every_config_is_within_bounds():
    for c in build_sweep(10):
        assert _within_bounds(c.genome), f"{c.label} out of bounds: {c.genome}"


def test_strategies_are_self_healing_tiers():
    # low does not read notes/observations, so the sweep never uses it.
    assert {c.strategy for c in build_sweep(10)} <= {"medium", "high"}


def test_varied_configs_actually_differ_from_default():
    configs = build_sweep(10)
    varied = configs[1:]
    assert all(c.diffs() for c in varied), "non-control configs must change at least one param"
    # and the diffs only touch real genome keys
    for c in varied:
        assert set(c.diffs()).issubset(DEFAULT_PARAMS)


def test_build_sweep_extends_past_designed_set():
    configs = build_sweep(12)
    assert len(configs) == 12
    assert len({c.label for c in configs}) == 12
    assert all(_within_bounds(c.genome) for c in configs)
    # beyond the designed 10, labels are deterministic perturbations
    assert configs[10].label.startswith("perturb-")


def test_build_sweep_is_deterministic():
    a = build_sweep(12)
    b = build_sweep(12)
    assert [(c.label, c.genome, c.strategy) for c in a] == [
        (c.label, c.genome, c.strategy) for c in b
    ]


def test_build_sweep_rejects_zero():
    with pytest.raises(ValueError):
        build_sweep(0)


def _entry(label, *, won=False, reached=False, pewter=False, maps=1, lead=0, turns=100) -> dict:
    return {
        "label": label,
        "reached_pewter": pewter,
        "maps_visited": maps,
        "total_turns": turns,
        "brock": {"won": won, "reached": reached, "lead_level": lead, "turns": None},
        "captured_state": None,
    }


def test_rank_sweep_prioritises_win_then_progress():
    entries = [
        _entry("lost-far", won=False, reached=True, maps=6, lead=11),
        _entry("won", won=True, reached=True, maps=5, lead=12),
        _entry("stuck", won=False, reached=False, maps=2, lead=0),
        _entry("explored", won=False, reached=False, maps=4, lead=0),
    ]
    ranked = [e["label"] for e in rank_sweep(entries)]
    assert ranked[0] == "won"  # a win beats everything
    assert ranked[1] == "lost-far"  # reached Brock beats not-reached
    assert ranked.index("explored") < ranked.index("stuck")  # more maps explored ranks higher


def test_rank_sweep_breaks_ties_on_lead_level():
    entries = [
        _entry("low-lead", reached=True, maps=5, lead=9),
        _entry("high-lead", reached=True, maps=5, lead=14),
    ]
    ranked = [e["label"] for e in rank_sweep(entries)]
    assert ranked == ["high-lead", "low-lead"]


def test_build_report_best_is_top_ranked():
    entries = [
        _entry("a", maps=2),
        _entry("b", won=True, reached=True, maps=5, lead=12),
    ]
    report = build_report(entries, n=2, max_turns=2500, seed_state="./states/route1.state")
    assert report["n"] == 2
    assert report["max_turns"] == 2500
    assert report["best"]["label"] == "b"
    assert report["entries"][0]["label"] == "b"
