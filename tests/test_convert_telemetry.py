"""Tests for the telemetry -> SFT corpus converter."""

from pathlib import Path

from autotune.convert_telemetry import chat, load_events

FIXTURES = Path(__file__).parent / "fixtures" / "convert"


def test_load_events_parses_and_counts_skipped():
    events, skipped = load_events([FIXTURES])
    assert skipped == 1
    types = [e["event_type"] for e in events]
    assert types == ["battle_outcome", "move_result", "milestone"]
    assert all(e["_file"] == "2026-06-28" for e in events)


def test_chat_shape():
    ex = chat("sys", "usr", "ans", "battle-outcome")
    assert ex["domain"] == "battle-outcome"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == "ans"
