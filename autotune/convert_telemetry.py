"""Convert pokemon.game.v1 telemetry into a multi-domain SFT corpus.

Deterministic (seeded) converter per
docs/superpowers/specs/2026-07-05-telemetry-sft-converter-design.md.
Five generators (battle-outcome, move-choice, battle-action, genome, narrator) feed one dedup +
balance + stratified-split assembly. Pure Python; unit-tested; run via
``uv run python -m autotune.convert_telemetry``.

Memory: every input file and pipeline phase is logged to ``<out>/memlog.jsonl`` (and stderr) as
it happens, and the run aborts itself with exit code 3 once RSS passes ``--max-rss-gb``
(default: ``AUTOTUNE_MAX_RSS_GB`` or half of MemTotal). See ``autotune.memlog``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

from autotune.memlog import GB, MemLog, MemoryBudgetExceeded, default_budget_bytes

PROGRESS_EVERY = 500_000  # lines between in-file memory checkpoints
EXIT_MEMORY_BUDGET = 3

# The event types a generator below actually reads. pokemon-kafka's data/game sink is a
# 21 GB firehose in which ``decision`` / ``agent_state`` / ``stuck`` / ``overworld`` lines
# outnumber these ~7:1; parsing them into dicts is what OOM-killed the box (see memlog.py).
WANTED_EVENT_TYPES = frozenset(
    {
        "battle_outcome",
        "move_result",
        "battle",
        "milestone",
        "map_change",
        "discovery",
        "battle_end",
    }
)
_EVENT_TYPE_RE = re.compile(r'"event_type":\s*"([^"]*)"')
# Event time on the raw line. File names are sink dates, not event dates (a file dated 08-22
# carries events from 06-29), so vintage is always the event's own stamp.
_OCCURRED_RE = re.compile(r'"(?:occurred_at|ts)":\s*"(\d{4}-\d{2}-\d{2})')


def before_since(line: str, since: str | None) -> bool:
    """True when the raw line's event date is older than ``since`` (YYYY-MM-DD)."""
    if not since:
        return False
    m = _OCCURRED_RE.search(line)
    return bool(m) and m.group(1) < since


def load_events(
    roots: list[Path], memlog: MemLog | None = None, since: str | None = None
) -> tuple[list[dict], int]:
    """Read every *.jsonl under each root. Returns (events, skipped_line_count).

    A line whose ``event_type`` is not in ``WANTED_EVENT_TYPES``, or whose event date is before
    ``since``, is dropped on the raw text before ``json.loads``; lines without an ``event_type``
    key take the old path (parsed, kept if they carry ``event_type`` or ``type``). With
    ``memlog``, one record per file (bytes on disk, lines, dropped, kept, rss after) and one
    every ``PROGRESS_EVERY`` lines inside a file; each record checks the RSS budget.
    """
    events: list[dict] = []
    skipped = 0
    for root in roots:
        if not root.exists():
            print(f"[convert] warning: missing data root {root}")
            continue
        root_files = root_kept = 0
        for path in sorted(root.rglob("*.jsonl")):
            lines = kept = dropped = 0
            with open(path) as f:
                for line in f:
                    lines += 1
                    if memlog and lines % PROGRESS_EVERY == 0:
                        memlog.log(
                            "load_events.progress",
                            file=str(path),
                            lines=lines,
                            dropped=dropped,
                            kept=kept,
                        )
                    m = _EVENT_TYPE_RE.search(line)
                    if (m and m.group(1) not in WANTED_EVENT_TYPES) or before_since(line, since):
                        dropped += 1
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if not isinstance(d, dict):
                        skipped += 1  # a bare scalar/list line is not an event
                        continue
                    if "event_type" in d or "type" in d:
                        d["_file"] = path.stem
                        events.append(d)
                        kept += 1
            root_files += 1
            root_kept += kept
            if memlog:
                memlog.note_file(str(path), kept)
                memlog.log(
                    "load_events.file",
                    file=str(path),
                    bytes=path.stat().st_size,
                    lines=lines,
                    dropped=dropped,
                    kept=kept,
                    events=len(events),
                )
        if memlog:
            memlog.log("load_events.root", root=str(root), files=root_files, kept=root_kept)
    return events, skipped


def provenance(event: dict | None) -> dict:
    """Where a row came from: source file, event time, run, game. Missing fields are omitted."""
    if not event:
        return {}
    meta = {
        "file": event.get("_file"),
        "occurred_at": event.get("occurred_at") or event.get("ts"),
        "run_id": event.get("run_id"),
        "game": event.get("game"),
    }
    return {k: v for k, v in meta.items() if v}


def chat(system: str, user: str, assistant: str, domain: str, event: dict | None = None) -> dict:
    """Build one chat-format SFT example; ``event`` stamps its provenance as ``meta``."""
    row = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "domain": domain,
    }
    meta = provenance(event)
    if meta:
        row["meta"] = meta
    return row


# Telemetry events stamped by pokemon-kafka carry a "game" field; legacy files predate
# it, and everything recorded before the field existed was a Pokemon Red run.
GAME_LABELS = {"red_blue": "Red/Blue", "yellow": "Yellow"}


def game_label(event: dict | None) -> str:
    """Human game name for prompts ("Red" for legacy events without the field)."""
    return GAME_LABELS.get((event or {}).get("game", ""), "Red")


def battle_system(label: str) -> str:
    return (
        f"You are the battle advisor for a Pokemon {label} agent. "
        "Answer with only the requested JSON."
    )


BATTLE_SYSTEM = battle_system("Red")  # legacy default, kept for reference


def gen_battle_outcome(events: list[dict]) -> list[dict]:
    """battle_outcome rows -> win prediction + fight/flee recommendation examples."""
    out = []
    for e in events:
        if e.get("event_type") != "battle_outcome":
            continue
        d = e["data"]
        moves = ", ".join(d["user_move_types"])
        user = (
            f"Battle start.\n"
            f"Your Pokemon: {d['user_species']} (lv {d['user_level']}, "
            f"HP {d['user_hp_start']}/{d['user_max_hp']}), move types: {moves}.\n"
            f"Enemy: {d['enemy_species']} (lv {d['enemy_level']}, "
            f"{d['enemy_type']} type). "
            f"Level gap: {d['level_gap']:+d}. "
            f"Healing available: {'yes' if d['had_healing'] else 'no'}.\n"
            "Will the agent win this battle, and should it fight or flee? "
            'Respond with JSON {"win": bool, "recommendation": "fight"|"flee"}.'
        )
        answer = json.dumps({"win": d["won"], "recommendation": "fight" if d["won"] else "flee"})
        out.append(chat(battle_system(game_label(e)), user, answer, "battle-outcome", e))
    return out


def damage_bucket(damage: int, enemy_max_hp: int, one_shot: bool) -> str:
    """Bucket damage as a fraction of enemy max HP.

    Buckets: none / <15% / 15-40% / >40% or one-shot.
    """
    if one_shot:
        return "heavy"
    if damage <= 0:
        return "none"
    frac = damage / max(enemy_max_hp, 1)
    if frac < 0.15:
        return "weak"
    if frac <= 0.40:
        return "solid"
    return "heavy"


def gen_move_choice(events: list[dict]) -> list[dict]:
    """move_result rows -> per-row damage buckets + aggregated best-move picks."""
    out = []
    by_matchup: dict[tuple[str, str], dict[str, list[int]]] = {}
    matchup_labels: dict[tuple[str, str], str] = {}
    matchup_latest: dict[tuple[str, str], dict] = {}
    for e in events:
        if e.get("event_type") != "move_result":
            continue
        d = e["data"]
        bucket = damage_bucket(d["damage_dealt"], d["enemy_max_hp"], d.get("one_shot", False))
        user = (
            f"{d['user_species']} (lv {d['user_level']}) uses {d['move']} "
            f"({d['move_type']}, power {d['move_power']}) against {d['enemy_species']} "
            f"(lv {d['enemy_level']}, {d['enemy_type']} type) with "
            f"{d['enemy_hp_before']}/{d['enemy_max_hp']} HP.\n"
            "How much damage relative to the enemy's max HP? "
            'Respond with JSON {"bucket": "none"|"weak"|"solid"|"heavy"}.'
        )
        out.append(
            chat(
                battle_system(game_label(e)), user, json.dumps({"bucket": bucket}), "move-choice", e
            )
        )
        key = (d["user_species"], d["enemy_type"])
        matchup_labels.setdefault(key, game_label(e))
        # the aggregate row is stamped with its newest contributing event
        latest = matchup_latest.get(key)
        if latest is None or (e.get("occurred_at") or "") > (latest.get("occurred_at") or ""):
            matchup_latest[key] = e
        by_matchup.setdefault(key, {}).setdefault(f"{d['move']} ({d['move_type']})", []).append(
            d["damage_dealt"]
        )
    for (species, enemy_type), moves in sorted(by_matchup.items()):
        if len(moves) < 2:
            continue
        means = {m: sum(v) / len(v) for m, v in moves.items()}
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        if ranked[0][1] == ranked[1][1]:
            continue  # tie: no ground-truth winner
        moves_desc = ", ".join(sorted(means))
        best_name = ranked[0][0].split(" (")[0]
        user = (
            f"{species} is fighting a {enemy_type}-type enemy. "
            f"Observed moves: {moves_desc}.\n"
            'Which move deals the most damage? Respond with JSON {"move": "..."}.'
        )
        out.append(
            chat(
                battle_system(matchup_labels[(species, enemy_type)]),
                user,
                json.dumps({"move": best_name}),
                "move-choice",
                matchup_latest[(species, enemy_type)],
            )
        )
    return out


def group_battles(events: list[dict]) -> list[tuple[list[dict], dict]]:
    """Per source file, attach each run of `battle` turns to the next `battle_outcome`."""
    groups: list[tuple[list[dict], dict]] = []
    by_file: dict[str, list[dict]] = {}
    for e in events:
        if e.get("event_type") in ("battle", "battle_outcome"):
            by_file.setdefault(e["_file"], []).append(e)
    for _file, evs in sorted(by_file.items()):
        evs = sorted(evs, key=lambda e: e.get("turn", 0))
        pending: list[dict] = []
        for e in evs:
            if e["event_type"] == "battle":
                pending.append(e)
            else:  # battle_outcome closes the current battle
                if pending:
                    groups.append((pending, e["data"]))
                pending = []
    return groups


def gen_battle_action(events: list[dict], rng: random.Random, cap: int = 800) -> list[dict]:
    """Turns of won battles -> state -> action examples (rejection sampling on outcome).

    Some sinks record a lean ``battle`` turn (HP and action only, no species or level); the
    prompt names both, so those turns are skipped and counted on stderr.
    """
    out = []
    lean = 0
    for turns, outcome in group_battles(events):
        if not outcome.get("won"):
            continue
        for e in turns:
            d = e["data"]
            try:
                action = json.loads(d["action"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            try:
                user = (
                    f"In battle: your Pokemon {d['player_species']} (lv {d['player_level']}) "
                    f"HP {d['player_hp']}/{d['player_max_hp']}; enemy {d['enemy_species']} "
                    f"(lv {d['enemy_level']}) HP {d['enemy_hp']}/{d['enemy_max_hp']}.\n"
                    "Choose the next action. Respond with the action JSON, e.g. "
                    '{"action": "fight", "move": "..."} or {"action": "run"}.'
                )
            except KeyError:
                lean += 1
                continue
            out.append(
                chat(battle_system(game_label(e)), user, json.dumps(action), "battle-action", e)
            )
    if lean:
        print(
            f"[convert] battle-action: skipped {lean} turns without species/level", file=sys.stderr
        )
    if len(out) > cap:
        out = rng.sample(out, cap)
    return out


def genome_system(label: str) -> str:
    return f"You tune a Pokemon {label} agent's survival genome. Respond with only the genome JSON."


GENOME_SYSTEM = genome_system("Red")  # legacy default, kept for reference


def gen_genome(rollout_roots: list[Path], label: str = "Red") -> list[dict]:
    """Above-median rollout genomes per scenario -> fitness-summary -> genome examples."""
    out = []
    for root in rollout_roots:
        if not root.exists():
            print(f"[convert] warning: missing rollout root {root}")
            continue
        for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rollouts = []
            for rdir in sorted(scenario_dir.glob("rollout-*")):
                gpath, fpath = rdir / "genome.json", rdir / "fitness.json"
                if not (gpath.exists() and fpath.exists()):
                    continue
                genome = json.loads(gpath.read_text())
                fitness = json.loads(fpath.read_text())
                rollouts.append((genome, fitness))
            if not rollouts:
                continue
            median_won = statistics.median(f.get("battles_won", 0) for _, f in rollouts)
            for genome, fitness in rollouts:
                if fitness.get("battles_won", 0) < median_won:
                    continue
                turns = fitness.get("turns", 0)
                battles = fitness.get("battles_won", 0)
                maps = fitness.get("maps_visited", 0)
                user = (
                    f"Scenario: {scenario_dir.name}. A rollout with this genome survived "
                    f"{turns} turns, won {battles} battles, "
                    f"and visited {maps} maps.\n"
                    "Propose the genome JSON that achieved this."
                )
                row = chat(genome_system(label), user, json.dumps(genome, sort_keys=True), "genome")
                row["meta"] = {"file": scenario_dir.name, "rollout": rdir.name}
                out.append(row)
    return out


def narrator_system(label: str) -> str:
    return (
        f"You are the live commentator for an autonomous Pokemon {label} run. "
        "Reply with one short sentence."
    )


NARRATOR_SYSTEM = narrator_system("Red")  # legacy default, kept for reference

NARRATOR_TEMPLATES: dict[str, list[str]] = {
    "milestone": [
        "Huge moment: {description}",
        "The run just hit a milestone — {description}",
        "Checkpoint reached: {description}",
        "That's the milestone the chat was waiting for: {description}",
        "Progress locked in: {description}",
    ],
    "map_change": [
        "The agent crosses from {from_map} into {to_map}.",
        "New area: leaving {from_map}, entering {to_map}.",
        "Transition — {from_map} is behind us, {to_map} ahead.",
        "The party steps out of {from_map} and into {to_map}.",
        "Map change: {from_map} to {to_map}.",
    ],
    "discovery": [
        "Found something: {text}",
        "The agent uncovers a clue — {text}",
        "On-screen text spotted: {text}",
        "A discovery in the overworld: {text}",
        "New info just dropped: {text}",
    ],
    "battle_end": [
        "Battle over — the agent {result} against {enemy_species}.",
        "That fight with {enemy_species} ends: {result}.",
        "Result vs {enemy_species}: {result}.",
        "The {enemy_species} encounter wraps up — {result}.",
        "Dust settles on the {enemy_species} battle: {result}.",
    ],
}


def gen_narrator(events: list[dict], rng: random.Random) -> list[dict]:
    """Notable events -> one-sentence play-by-play from seeded template pools."""
    out = []
    for e in events:
        etype = e.get("event_type")
        if etype not in NARRATOR_TEMPLATES:
            continue
        d = dict(e.get("data", {}))
        d.setdefault("description", "")
        d.setdefault("from_map", "the last area")
        d.setdefault("to_map", f"map {d.get('map_id', '?')}")
        d.setdefault("text", d.get("kind", ""))
        d.setdefault("result", "is decided")
        d.setdefault("enemy_species", "the enemy")
        template = rng.choice(NARRATOR_TEMPLATES[etype])
        try:
            sentence = template.format(**d)
        except (KeyError, IndexError):
            continue
        user = (
            "Narrate this game event for the stream overlay in one sentence:\n"
            f"{json.dumps(e.get('data', {}), sort_keys=True)}"
        )
        out.append(chat(narrator_system(game_label(e)), user, sentence, "narrator", e))
    return out


DOMAINS = (
    "battle-outcome",
    "move-choice",
    "battle-action",
    "genome",
    "narrator",
    "handoff",
    "puzzle-consult",
    "gate-text",
)


# A species the sink could not name is written as "#" + its hex id ("#6B"). The species table
# was fixed on 2026-08-26; rows built from earlier events can still carry these, and a prompt
# that names a Pokemon by an unresolved id teaches nothing. Dropped here, counted per domain.
UNRESOLVED_SPECIES_RE = re.compile(r"(?<![\w#])#[0-9A-F]{2}(?![\w])")


def drop_unresolved(examples: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Drop rows whose prompt names an unresolved species. Returns (kept, dropped_by_domain)."""
    kept, dropped = [], {}
    for ex in examples:
        if UNRESOLVED_SPECIES_RE.search(ex["messages"][1]["content"]):
            dropped[ex["domain"]] = dropped.get(ex["domain"], 0) + 1
        else:
            kept.append(ex)
    return kept, dropped


def _key(ex: dict) -> str:
    payload = ex["messages"][1]["content"] + "\x00" + ex["messages"][2]["content"]
    return hashlib.sha256(payload.encode()).hexdigest()


def dedupe(examples: list[dict]) -> list[dict]:
    """Drop exact (user, assistant) duplicates, first-seen wins."""
    seen, out = set(), []
    for ex in examples:
        k = _key(ex)
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return out


def balance(examples: list[dict], rng: random.Random, max_frac: float = 0.4) -> list[dict]:
    """Down-sample any domain above max_frac of the corpus (seeded).

    Domains that already sit at or under max_frac of the *original* total are
    left alone entirely -- they never become shrink candidates, no matter how
    their share of the corpus shifts as other domains get resampled. Domains
    that started over max_frac are down-sampled to fixed point: on each pass,
    recompute the cap against the current (possibly already-shrunk) total,
    pick the largest domain -- among the originally-over-cap set -- that is
    still over that cap, and resample it down to the size that makes it
    exactly max_frac of the corpus once combined with every other domain's
    current size. Repeat until every originally-over-cap domain fits.

    Recomputing against the shrinking total (rather than the original one) is
    what makes this converge correctly when two or more domains are
    simultaneously over cap: each domain's target reflects every other
    domain's *current* size, not its original one. Restricting the candidate
    set to the domains that were over cap *originally* is what stops the
    cascade from reaching back and shrinking domains that were never
    overrepresented -- e.g. a small domain whose share balloons only because
    the dominant domain next to it got smaller.
    """
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    if len(by_domain) > 1 and 0 < max_frac < 1:
        total = sum(len(v) for v in by_domain.values())
        over_cap = {d for d, rows in by_domain.items() if len(rows) > max_frac * total}
        while over_cap:
            total = sum(len(v) for v in by_domain.values())
            still_over = [d for d in over_cap if len(by_domain[d]) > max_frac * total]
            if not still_over:
                break
            domain = max(still_over, key=lambda d: len(by_domain[d]))
            rows = by_domain[domain]
            rest = total - len(rows)
            target = int(max_frac / (1 - max_frac) * rest)
            if rest > 0:
                target = max(target, 1)
            if target >= len(rows):
                break  # safety net; shouldn't trigger given the over-cap check above
            by_domain[domain] = rng.sample(rows, target)
    out = [ex for d in sorted(by_domain) for ex in by_domain[d]]
    return out


def split(examples: list[dict], rng: random.Random, valid_frac: float = 0.1) -> tuple[list, list]:
    """Stratified train/valid split, deterministic under a fixed rng."""
    train, valid = [], []
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    for domain in sorted(by_domain):
        rows = by_domain[domain][:]
        rng.shuffle(rows)
        n_valid = max(1, int(len(rows) * valid_frac)) if len(rows) >= 2 else 0
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    return train, valid


def write_corpus(
    out_dir: Path,
    examples: list[dict],
    train: list[dict],
    valid: list[dict],
    skipped: int,
    seed: int,
) -> dict:
    """Write corpus/train/valid/stats files; returns the stats dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_text = "\n".join(json.dumps(ex, sort_keys=True) for ex in examples) + "\n"
    (out_dir / "corpus.jsonl").write_text(corpus_text)
    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps({"messages": ex["messages"]}, sort_keys=True) + "\n")
    with open(out_dir / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    stamped: list[str] = []
    for ex in examples:
        counts[ex["domain"]] = counts.get(ex["domain"], 0) + 1
        when = ex.get("meta", {}).get("occurred_at")
        if when:
            stamped.append(when[:10])
    stats = {
        "total": len(examples),
        "train": len(train),
        "valid": len(valid),
        "domains": counts,
        "skipped_lines": skipped,
        "seed": seed,
        "corpus_sha256": hashlib.sha256(corpus_text.encode()).hexdigest(),
        "vintage": {
            "min": min(stamped) if stamped else None,
            "max": max(stamped) if stamped else None,
            "stamped": len(stamped),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert pokemon.game.v1 telemetry into an SFT corpus.")
    p.add_argument("--pk-data", type=Path, default=Path("../pokemon-kafka/data"))
    p.add_argument("--ee-data", type=Path, default=Path("data/telemetry"))
    p.add_argument(
        "--rollouts",
        type=Path,
        action="append",
        default=None,
        help="rollout roots (repeatable); default: out/rollouts out/harvest",
    )
    p.add_argument(
        "--pk-root",
        type=Path,
        default=Path("../pokemon-kafka"),
        help="pokemon-kafka checkout: its data/telemetry/game sink and docs/learnings "
        "feed the handoff domains",
    )
    p.add_argument(
        "--resolutions",
        type=Path,
        default=Path(__file__).with_name("handoff_resolutions.json"),
        help="curated operator resolutions for the handoff domain",
    )
    p.add_argument("--out", type=Path, default=Path("data/sft_v4"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="drop events stamped before this date (the later sinks carry the fuller schema)",
    )
    p.add_argument("--min-total", type=int, default=500)
    p.add_argument("--action-cap", type=int, default=800)
    p.add_argument(
        "--mem-log",
        type=Path,
        default=None,
        help="memory trail (jsonl, appended as it happens); default: <out>/memlog.jsonl",
    )
    p.add_argument(
        "--max-rss-gb",
        type=float,
        default=None,
        help="abort (exit 3) once RSS passes this; 0 disables. "
        "Default: $AUTOTUNE_MAX_RSS_GB, else half of MemTotal",
    )
    p.add_argument(
        "--game",
        choices=["red", "red_blue", "yellow"],
        default="red",
        help="Game label for genome examples (rollout artifacts carry no game field; "
        "telemetry-derived examples label themselves from each event)",
    )
    args = p.parse_args(argv)
    genome_label = {"red": "Red", **GAME_LABELS}[args.game]
    limit = default_budget_bytes() if args.max_rss_gb is None else int(args.max_rss_gb * GB)
    memlog = MemLog(args.mem_log or args.out / "memlog.jsonl", limit_bytes=limit, label="convert")

    try:
        return _run(args, rng=random.Random(args.seed), genome_label=genome_label, memlog=memlog)
    except MemoryBudgetExceeded as exc:
        memlog.stream = None  # the abort record is written; don't echo it twice
        memlog.log("abort", check=False, error=str(exc))
        print(f"[convert] FATAL: {exc}", file=sys.stderr)
        print(f"[convert] memory trail: {memlog.path}", file=sys.stderr)
        for name, kept in memlog.top_files():
            print(f"[convert]   {kept:>10d} events kept from {name}", file=sys.stderr)
        print(
            "[convert] raise --max-rss-gb (or AUTOTUNE_MAX_RSS_GB) to allow more, "
            "or narrow --pk-data/--ee-data to the files that carry training events",
            file=sys.stderr,
        )
        return EXIT_MEMORY_BUDGET


def _run(args, rng: random.Random, genome_label: str, memlog: MemLog) -> int:
    events, skipped = load_events([args.pk_data, args.ee_data], memlog=memlog, since=args.since)
    memlog.log("load_events", events=len(events), skipped=skipped)
    rollout_roots = args.rollouts or [Path("out/rollouts"), Path("out/harvest")]

    from autotune import handoff_corpus as ho

    expedition, ho_skipped = ho.load_expedition_events(
        args.pk_root / "data" / "telemetry" / "game", memlog=memlog, since=args.since
    )
    memlog.log("load_expedition_events", events=len(expedition), skipped=ho_skipped)
    learnings = args.pk_root / "docs" / "learnings"
    handoffs, unmatched = ho.gen_handoff(
        expedition, learnings, ho.load_resolutions(args.resolutions)
    )
    memlog.log("gen_handoff", examples=len(handoffs), unmatched=unmatched)

    examples: list[dict] = []
    generators = (
        ("gen_battle_outcome", lambda: gen_battle_outcome(events)),
        ("gen_move_choice", lambda: gen_move_choice(events)),
        ("gen_battle_action", lambda: gen_battle_action(events, rng, cap=args.action_cap)),
        ("gen_genome", lambda: gen_genome(rollout_roots, label=genome_label)),
        ("gen_narrator", lambda: gen_narrator(events, rng)),
        ("handoffs", lambda: handoffs),
        ("gen_puzzle_consult", lambda: ho.gen_puzzle_consult(expedition, learnings)),
        ("gen_gate_text", lambda: ho.gen_gate_text(expedition)),
    )
    for name, gen in generators:
        rows = gen()
        examples += rows
        memlog.log(name, examples=len(rows), total=len(examples))
    examples, unresolved = drop_unresolved(examples)
    memlog.log("drop_unresolved", total=len(examples), dropped=sum(unresolved.values()))
    examples = dedupe(examples)
    memlog.log("dedupe", total=len(examples))
    examples = balance(examples, rng)
    memlog.log("balance", total=len(examples))
    train, valid = split(examples, rng)
    memlog.log("split", train=len(train), valid=len(valid))
    stats = write_corpus(args.out, examples, train, valid, skipped + ho_skipped, args.seed)
    memlog.log("write_corpus", total=stats["total"])
    stats["handoffs_unmatched"] = unmatched
    stats["since"] = args.since
    stats["dropped_unresolved_species"] = unresolved
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    from autotune.dataset_card import write_card

    write_card(args.out, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))

    empty = [d for d in DOMAINS if stats["domains"].get(d, 0) == 0]
    if empty:
        print(f"[convert] FATAL: empty domains: {empty}", file=sys.stderr)
        return 1
    if stats["total"] < args.min_total:
        msg = f"[convert] FATAL: only {stats['total']} examples (< {args.min_total})"
        print(msg, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
