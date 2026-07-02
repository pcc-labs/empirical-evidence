"""Union domain-tagged SFT corpora into one training set.

Each harvest persists its full tagged corpus as ``corpus.jsonl`` (rows:
``{"messages": [...], "domains": [...]}``) — the forest harvest contributes battle/discovery
pairs, the map-grained harvest contributes nav pairs. This CLI unions those corpora, fails
LOUDLY when an input is missing or empty (a silent empty union would train the LoRA on nothing),
prints the per-domain census, and writes the merged tagged ``corpus.jsonl`` plus the stripped
``train.jsonl`` / ``valid.jsonl`` that ``train_sft`` consumes.

``load_corpus`` / ``merge_corpora`` / ``domain_census`` are pure and unit-tested; the CLI wrapper
is exercised by the smoke run (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotune.nudge_sft import write_corpus, write_sft_data
from autotune.train_sft import load_jsonl


def load_corpus(path: Path) -> list[dict]:
    """Read one tagged corpus JSONL; SystemExit on missing/empty (loud, per issue #10)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[merge_corpus] {path} does not exist — run its harvest first.")
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"[merge_corpus] {path} is empty — its harvest found no gradient.")
    return rows


def merge_corpora(corpora: list[list[dict]]) -> list[dict]:
    """Union corpora in order, dropping exact-duplicate examples (same messages), keeping first."""
    seen: set[str] = set()
    merged: list[dict] = []
    for rows in corpora:
        for row in rows:
            key = json.dumps(row.get("messages"), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def domain_census(examples: list[dict]) -> dict[str, int]:
    """Examples per domain tag (an example with N tags counts once per tag; no tag = untagged)."""
    census: dict[str, int] = {}
    for ex in examples:
        for domain in ex.get("domains") or ["untagged"]:
            census[domain] = census.get(domain, 0) + 1
    return census


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Union domain-tagged SFT corpora for train_sft.")
    p.add_argument("--inputs", nargs="+", required=True, help="corpus.jsonl paths to union")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    merged = merge_corpora([load_corpus(Path(x)) for x in args.inputs])
    census = domain_census(merged)
    corpus_path = write_corpus(args.out_dir / "corpus.jsonl", merged)
    train_path, _valid_path = write_sft_data(args.out_dir, merged, args.valid_frac, args.seed)
    print(f"[merge_corpus] {len(merged)} examples, census {json.dumps(census)}")
    print(f"[merge_corpus] tagged corpus -> {corpus_path}; train data -> {train_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
