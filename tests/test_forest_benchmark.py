"""Tests for the per-domain forest checkpoint benchmark's pure seams."""

from __future__ import annotations

import json

from autotune.forest_benchmark import (
    ForestBenchRow,
    format_forest_trend,
    propose_forest_genome,
    summarize_verdicts,
)
from autotune.forest_story import score_forest


def _verdict(events):
    return score_forest(events)


_ENTER = {"event_type": "overworld", "turn": 0, "data": {"map_id": 51}}
_TWIN = {"event_type": "battle_outcome", "turn": 1, "data": {"battle_type": 2, "won": True}}
_EXIT = {"event_type": "overworld", "turn": 2, "data": {"map_id": 13}}


def test_propose_forest_genome_merges_parsed_over_params():
    def proposer(prompt):
        assert "Viridian Forest" in prompt
        return json.dumps({"hp_run_threshold": 0.15})

    genome, parsed = propose_forest_genome(proposer, {"hp_run_threshold": 0.6}, _verdict([_ENTER]))
    assert parsed and genome["hp_run_threshold"] == 0.15


def test_propose_forest_genome_falls_back_on_garbage():
    genome, parsed = propose_forest_genome(
        lambda _p: "not json at all", {"hp_run_threshold": 0.6}, _verdict([_ENTER])
    )
    assert not parsed and genome == {"hp_run_threshold": 0.6}


def test_summarize_verdicts_means_domains_across_states():
    row = summarize_verdicts("100", [_verdict([_ENTER, _TWIN, _EXIT]), _verdict([_ENTER])])
    assert row.label == "100"
    assert row.reward == 2.0  # (3 + 1) / 2
    assert row.domains == {"nav": 1.5, "battle": 0.5, "discovery": 0.0}
    assert row.crossed == 0.5


def test_format_forest_trend_has_domain_columns():
    baseline = ForestBenchRow(
        label="baseline", reward=2.0, domains={"nav": 1.0, "battle": 1.0, "discovery": 0.0},
        crossed=0.0, parsed=True,
    )
    row = ForestBenchRow(
        label="final", reward=4.0, domains={"nav": 2.0, "battle": 2.0, "discovery": 0.0},
        crossed=1.0, parsed=False,
    )
    table = format_forest_trend(baseline, [row])
    assert "nav" in table and "battle" in table and "discov" in table
    assert "baseline" in table and "final" in table
    assert "parse-fallback" in table  # unparsed rows are visibly flagged
