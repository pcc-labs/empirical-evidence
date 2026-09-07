from pathlib import Path

from autotune.convert_telemetry import DOMAINS
from autotune.dataset_card import SEATS, render_card, write_card


def test_every_domain_has_a_seat():
    assert {d for _, d, _ in SEATS} == set(DOMAINS)


def test_card_reflects_stats(tmp_path):
    stats = {
        "total": 10,
        "train": 9,
        "valid": 1,
        "seed": 42,
        "domains": {"gate-text": 4, "narrator": 6},
        "vintage": {"min": "2026-08-15", "max": "2026-09-05", "stamped": 10},
        "since": "2026-08-15",
        "skipped_lines": 2,
        "handoffs_unmatched": [{"map": 1}],
        "corpus_sha256": "abc123",
        "dropped_unresolved_species": {"battle-outcome": 3},
    }
    path = write_card(tmp_path, stats)
    text = path.read_text()
    assert path == tmp_path / "README.md"
    assert "| Forger | gate-text | 4 |" in text
    assert "2026-08-15 to 2026-09-05" in text
    assert "1 handoffs with no curated resolution" in text
    assert "--revision abc123" in text
    assert "(battle-outcome: 3)" in text
    assert render_card(stats, Path("x")).startswith("---\nlicense")
