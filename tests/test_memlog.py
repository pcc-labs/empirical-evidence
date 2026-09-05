"""Tests for the memory trail + RSS budget."""

import json

import pytest

from autotune.memlog import GB, MemLog, MemoryBudgetExceeded, default_budget_bytes, rss_bytes


def test_rss_is_measured():
    assert rss_bytes() > 0


def test_memlog_appends_records_as_they_happen(tmp_path):
    path = tmp_path / "trail" / "memlog.jsonl"
    m = MemLog(path, stream=None)
    m.note_file("a.jsonl", 3)
    m.log("phase-a", events=3)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["start", "phase-a"]
    assert rows[1]["events"] == 3
    assert rows[1]["rss_gb"] > 0 and rows[1]["peak_gb"] >= rows[1]["rss_gb"]
    assert "pid" in rows[0]
    assert m.count == 2
    assert m.top_files() == [("a.jsonl", 3)]


def test_budget_raises_and_records(tmp_path):
    path = tmp_path / "memlog.jsonl"
    # the start record itself would trip a 1-byte limit; construct, then tighten
    m = MemLog(path, stream=None)
    m.limit = 1
    with pytest.raises(MemoryBudgetExceeded) as ei:
        m.log("load")
    assert ei.value.phase == "load"
    assert ei.value.rss > 1
    assert json.loads(path.read_text().splitlines()[-1])["phase"] == "load"


def test_budget_zero_never_raises():
    m = MemLog(None, limit_bytes=0, stream=None)
    m.log("anything")


def test_default_budget_prefers_env(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_MAX_RSS_GB", "2.5")
    assert default_budget_bytes() == int(2.5 * GB)
