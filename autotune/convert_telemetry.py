"""Convert pokemon.game.v1 telemetry into a multi-domain SFT corpus.

Deterministic (seeded) converter per
docs/superpowers/specs/2026-07-05-telemetry-sft-converter-design.md.
Five generators (battle-outcome, move-choice, battle-action, genome, narrator) feed one dedup +
balance + stratified-split assembly. Pure Python; unit-tested; run via
``uv run python -m autotune.convert_telemetry``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path


def load_events(roots: list[Path]) -> tuple[list[dict], int]:
    """Read every *.jsonl under each root. Returns (events, skipped_line_count)."""
    events: list[dict] = []
    skipped = 0
    for root in roots:
        if not root.exists():
            print(f"[convert] warning: missing data root {root}")
            continue
        for path in sorted(root.rglob("*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if "event_type" in d or "type" in d:
                        d["_file"] = path.stem
                        events.append(d)
    return events, skipped


def chat(system: str, user: str, assistant: str, domain: str) -> dict:
    """Build one chat-format SFT example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "domain": domain,
    }


BATTLE_SYSTEM = (
    "You are the battle advisor for a Pokemon Red agent. Answer with only the requested JSON."
)


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
        out.append(chat(BATTLE_SYSTEM, user, answer, "battle-outcome"))
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
        out.append(chat(BATTLE_SYSTEM, user, json.dumps({"bucket": bucket}), "move-choice"))
        key = (d["user_species"], d["enemy_type"])
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
        out.append(chat(BATTLE_SYSTEM, user, json.dumps({"move": best_name}), "move-choice"))
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
    """Turns of won battles -> state -> action examples (rejection sampling on outcome)."""
    out = []
    for turns, outcome in group_battles(events):
        if not outcome.get("won"):
            continue
        for e in turns:
            d = e["data"]
            try:
                action = json.loads(d["action"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            user = (
                f"In battle: your Pokemon {d['player_species']} (lv {d['player_level']}) "
                f"HP {d['player_hp']}/{d['player_max_hp']}; enemy {d['enemy_species']} "
                f"(lv {d['enemy_level']}) HP {d['enemy_hp']}/{d['enemy_max_hp']}.\n"
                "Choose the next action. Respond with the action JSON, e.g. "
                '{"action": "fight", "move": "..."} or {"action": "run"}.'
            )
            out.append(chat(BATTLE_SYSTEM, user, json.dumps(action), "battle-action"))
    if len(out) > cap:
        out = rng.sample(out, cap)
    return out


GENOME_SYSTEM = "You tune a Pokemon Red agent's survival genome. Respond with only the genome JSON."


def gen_genome(rollout_roots: list[Path]) -> list[dict]:
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
                out.append(chat(GENOME_SYSTEM, user, json.dumps(genome, sort_keys=True), "genome"))
    return out


NARRATOR_SYSTEM = (
    "You are the live commentator for an autonomous Pokemon Red run. Reply with one short sentence."
)

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
        out.append(chat(NARRATOR_SYSTEM, user, sentence, "narrator"))
    return out


DOMAINS = ("battle-outcome", "move-choice", "battle-action", "genome", "narrator")


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

    Domains that already sit at or under max_frac of the original total are left
    untouched. Each domain that started over max_frac is down-sampled, in one
    pass, to the size that would make it exactly max_frac of the corpus once
    combined with the untouched domains. A naive re-check-everything loop
    cascades here: shrinking the dominant domain inflates every other domain's
    *share* of the new, smaller total, which can then trip the same cap and
    zero out domains that were never actually overrepresented.
    """
    by_domain: dict[str, list[dict]] = {}
    for ex in examples:
        by_domain.setdefault(ex["domain"], []).append(ex)
    total = sum(len(v) for v in by_domain.values())
    if len(by_domain) > 1 and 0 < max_frac < 1:
        for domain, rows in by_domain.items():
            if len(rows) <= max_frac * total:
                continue
            rest = total - len(rows)
            target = int(max_frac / (1 - max_frac) * rest)
            if target < len(rows):
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
    for ex in examples:
        counts[ex["domain"]] = counts.get(ex["domain"], 0) + 1
    stats = {
        "total": len(examples),
        "train": len(train),
        "valid": len(valid),
        "domains": counts,
        "skipped_lines": skipped,
        "seed": seed,
        "corpus_sha256": hashlib.sha256(corpus_text.encode()).hexdigest(),
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
    p.add_argument("--out", type=Path, default=Path("data/sft_v3"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-total", type=int, default=500)
    p.add_argument("--action-cap", type=int, default=800)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    events, skipped = load_events([args.pk_data, args.ee_data])
    rollout_roots = args.rollouts or [Path("out/rollouts"), Path("out/harvest")]

    examples = (
        gen_battle_outcome(events)
        + gen_move_choice(events)
        + gen_battle_action(events, rng, cap=args.action_cap)
        + gen_genome(rollout_roots)
        + gen_narrator(events, rng)
    )
    examples = dedupe(examples)
    examples = balance(examples, rng)
    train, valid = split(examples, rng)
    stats = write_corpus(args.out, examples, train, valid, skipped, args.seed)
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
