"""The dataset card written next to every corpus: which seat each domain trains, the vintage
of the rows, the known defects, and the upload command. Rendered from ``stats.json`` so the
card can never disagree with the corpus it sits beside.
"""

from __future__ import annotations

import json
from pathlib import Path

# Domain -> the crew seat it trains. The seats are pokemon-kafka's expedition crew (Point Man /
# Extractor / Wheelman) plus the one we are building toward: the Forger, who owns everything
# the game says back -- gate sentences, refusals, blockers to talk to.
SEATS: tuple[tuple[str, str, str], ...] = (
    ("Wheelman", "battle-outcome", "will this fight be won; fight or flee"),
    ("Wheelman", "move-choice", "damage bucket per move; best move per matchup"),
    ("Wheelman", "battle-action", "next action from a won battle's turns"),
    ("Extractor", "puzzle-consult", "menu choice at a wall, labelled by what the engine returned"),
    ("Forger", "gate-text", "the game's refusal sentence -> the gate class and what clears it"),
    (
        "Forger",
        "npc-dialogue",
        "where a body stands and what it said -> what it is and what the talk yields",
    ),
    ("Operator", "handoff", "the supervisor's exhaustion facts -> what the human measured and did"),
    ("Narrator", "narrator", "one-sentence play-by-play for the overlay"),
    ("Genome", "genome", "above-median rollout genomes per scenario"),
)

UPLOAD = "hf upload <org>/<dataset> {out} --repo-type dataset --revision {sha}"


def render_card(stats: dict, out_dir: Path) -> str:
    domains = stats.get("domains", {})
    vintage = stats.get("vintage") or {}
    sha = stats.get("corpus_sha256", "")
    unmatched = stats.get("handoffs_unmatched") or []
    unresolved = stats.get("dropped_unresolved_species") or {}
    unresolved_note = (
        ", ".join(f"{d}: {n}" for d, n in sorted(unresolved.items())) if unresolved else "none"
    )
    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- pokemon-kafka",
        "- sft",
        "---",
        "",
        "# pokemon-kafka SFT corpus",
        "",
        f"Deterministic conversion of pokemon-kafka telemetry (seed {stats.get('seed')}). "
        f"{stats.get('total', 0)} examples: {stats.get('train', 0)} train, "
        f"{stats.get('valid', 0)} valid.",
        "",
        "## Seats",
        "",
        "| seat | domain | rows | what the row teaches |",
        "|---|---|---:|---|",
    ]
    for seat, domain, teaches in SEATS:
        lines.append(f"| {seat} | {domain} | {domains.get(domain, 0)} | {teaches} |")
    lines += [
        "",
        "## Vintage",
        "",
        "Rows carry `meta.occurred_at` (the event's own stamp, never the sink file's date), "
        "`meta.file`, `meta.run_id` and `meta.game` where the source event had them. "
        "Later events are the better ones: the sinks grew species, level and run ids over time.",
        "",
        f"- events stamped: {vintage.get('stamped', 0)} of {stats.get('total', 0)}",
        f"- range: {vintage.get('min')} to {vintage.get('max')}",
        f"- `--since` filter applied: {stats.get('since') or 'none'}",
        "",
        "## Known defects",
        "",
        f"- {stats.get('skipped_lines', 0)} source lines skipped (unparseable or not an object)",
        f"- {len(unmatched)} handoffs with no curated resolution yet",
        "- battle turns recorded without species/level are skipped (count on the run's stderr)",
        "- rows whose prompt named a species by unresolved hex id (`#6B`) were dropped "
        f"before dedupe; the species table was fixed on 2026-08-26 ({unresolved_note})",
        "",
        "## Provenance",
        "",
        f"- corpus sha256: `{sha}`",
        "- built by `uv run python -m autotune.convert_telemetry` in empirical-evidence",
        "",
        "## Upload",
        "",
        "```",
        UPLOAD.format(out=out_dir, sha=sha[:12] or "<sha>"),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_card(out_dir: Path, stats: dict) -> Path:
    path = out_dir / "README.md"
    path.write_text(render_card(stats, out_dir))
    return path


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1])
    print(write_card(out, json.loads((out / "stats.json").read_text())))
