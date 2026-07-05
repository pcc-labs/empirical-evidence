"""Tests for the corpus union CLI's pure seams (autotune/merge_corpus.py)."""

from __future__ import annotations

import json

import pytest

from autotune.merge_corpus import domain_census, load_corpus, merge_corpora


def _ex(content: str, domains: list[str]) -> dict:
    return {"messages": [{"role": "user", "content": content}], "domains": domains}


def test_merge_unions_and_dedupes_identical_messages():
    a = [_ex("one", ["nav"]), _ex("two", ["battle"])]
    b = [_ex("two", ["battle"]), _ex("three", ["discovery"])]
    merged = merge_corpora([a, b])
    assert [e["messages"][0]["content"] for e in merged] == ["one", "two", "three"]


def test_census_counts_each_tag_and_untagged():
    merged = [_ex("a", ["nav", "battle"]), _ex("b", ["battle"]), {"messages": []}]
    assert domain_census(merged) == {"nav": 1, "battle": 2, "untagged": 1}


def test_load_corpus_missing_is_loud(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        load_corpus(tmp_path / "nope.jsonl")


def test_load_corpus_empty_is_loud(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="empty"):
        load_corpus(p)


def test_load_corpus_roundtrip(tmp_path):
    p = tmp_path / "corpus.jsonl"
    rows = [_ex("a", ["nav"])]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_corpus(p) == rows
