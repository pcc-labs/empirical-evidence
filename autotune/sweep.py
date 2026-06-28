"""Phase-A preparation sweep: run N pokemon-kafka agents in parallel over distinct
genome × strategy configs, and learn which prep config gets furthest toward an equipped Brock.

Where ``rollout.py`` replicates *one* genome k times for the story loop, ``sweep.py`` runs N
*distinct* configs once each, fully isolated, then ranks them by how far down the road to Pewter
each got. The levers are the "arrive equipped" knobs the journey actually controls: healing
aggressiveness (``hp_heal_threshold``), flee threshold (``hp_run_threshold``), nav un-stick
aggressiveness (``stuck_threshold`` / ``waypoint_skip_distance`` / ``bt_max_attempts``), and the
decision tier (``--strategy``; medium/high read ``notes.md``+``observations.md`` and self-heal).

Isolation is what makes 10 concurrent emulators safe: each run gets its **own ROM copy** (so its
``<rom>.gb.ram`` shutdown write can't race another run's, and a stale ``.ram`` can't break a fresh
intro), its own telemetry dir, and its own captured ``pre_brock.state``. Any captured state is
copied into ``states/brock/`` so Phase B (``autotune.loop --mode brock``) can optimise the battle
genome against it.

``build_sweep`` / ``rank_sweep`` / ``build_report`` are pure and unit-tested; ``run_sweep`` is a
subprocess driver exercised by the real run (omitted from coverage, like rollout.py / speedrun.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from autotune.config import Config, load_config
from autotune.genome import DEFAULT_PARAMS, PARAM_BOUNDS, clamp_params
from autotune.speedrun import PathStudy, build_path_study
from autotune.story import load_routes
from autotune.verifier import extract_visited_maps, load_fitness, load_game_events

# A run to ~2500 turns can take a while; give each emulator a generous ceiling.
_SWEEP_TIMEOUT_S = 3600


# ---------------------------------------------------------------------------
# The designed sweep (pure)
# ---------------------------------------------------------------------------

# (label, genome overrides vs DEFAULT_PARAMS, strategy tier, one-line rationale). Index 0 is the
# control (default genome). Each row isolates or combines a single "arrive equipped" hypothesis.
_DESIGN: list[tuple[str, dict, str, str]] = [
    ("control", {}, "medium", "baseline: default genome, self-healing tier"),
    ("heal-0.40", {"hp_heal_threshold": 0.40}, "medium", "heal sooner to survive the forest"),
    ("heal-0.55", {"hp_heal_threshold": 0.55}, "medium", "heal very early (max survival)"),
    ("stand-ground", {"hp_run_threshold": 0.10}, "medium", "flee less -> grind more XP"),
    (
        "grind+heal",
        {"hp_run_threshold": 0.10, "hp_heal_threshold": 0.45},
        "medium",
        "stand and grind, but heal early",
    ),
    (
        "unstick-fast",
        {"stuck_threshold": 5, "waypoint_skip_distance": 5},
        "medium",
        "skip waypoints sooner at the choke",
    ),
    (
        "backtrack-more",
        {"bt_max_attempts": 5, "bt_restore_threshold": 10},
        "medium",
        "retry/backtrack harder when stuck",
    ),
    (
        "explorer",
        {
            "stuck_threshold": 5,
            "waypoint_skip_distance": 5,
            "bt_max_attempts": 5,
            "hp_heal_threshold": 0.45,
        },
        "medium",
        "aggressive nav + early heal combined",
    ),
    (
        "smart-grind",
        {"hp_heal_threshold": 0.40, "hp_run_threshold": 0.10},
        "high",
        "high tier (richer decisions) + grind/heal",
    ),
    (
        "smart-explorer",
        {
            "stuck_threshold": 5,
            "waypoint_skip_distance": 5,
            "bt_max_attempts": 5,
            "hp_heal_threshold": 0.45,
            "hp_run_threshold": 0.10,
        },
        "high",
        "high tier + full aggressive prep",
    ),
]


@dataclass(frozen=True)
class SweepConfig:
    """One prep config: a labelled genome × strategy with the hypothesis it tests."""

    label: str
    genome: dict
    strategy: str
    rationale: str

    def diffs(self) -> dict:
        """Genome fields that differ from the default (what this config actually changes)."""
        return {k: v for k, v in self.genome.items() if DEFAULT_PARAMS.get(k) != v}


def _perturb(rng: random.Random) -> dict:
    """A deterministic extra config (used only when n exceeds the designed set)."""
    overrides: dict = {}
    for key in rng.sample(
        ["hp_heal_threshold", "hp_run_threshold", "stuck_threshold", "waypoint_skip_distance"],
        k=2,
    ):
        lo, hi, typ = PARAM_BOUNDS[key]
        overrides[key] = typ(rng.uniform(lo, hi)) if typ is float else rng.randint(int(lo), int(hi))
    return overrides


def build_sweep(n: int = 10, seed: int = 7) -> list[SweepConfig]:
    """Build ``n`` distinct, bounds-valid prep configs. The first up-to-10 come from the designed
    set; beyond that, deterministic perturbations of the default genome. Pure."""
    if n < 1:
        raise ValueError("n must be >= 1")
    configs: list[SweepConfig] = []
    for label, overrides, strategy, why in _DESIGN[:n]:
        genome = clamp_params({**DEFAULT_PARAMS, **overrides})
        configs.append(SweepConfig(label=label, genome=genome, strategy=strategy, rationale=why))
    rng = random.Random(seed)
    extra = 0
    while len(configs) < n:
        extra += 1
        overrides = _perturb(rng)
        genome = clamp_params({**DEFAULT_PARAMS, **overrides})
        strategy = rng.choice(["medium", "high"])
        configs.append(
            SweepConfig(
                label=f"perturb-{extra}",
                genome=genome,
                strategy=strategy,
                rationale="deterministic extra config beyond the designed set",
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Ranking + report (pure)
# ---------------------------------------------------------------------------


def _rank_key(entry: dict) -> tuple:
    """Sort key: won > reached-Brock/Pewter > more maps explored > higher lead level.

    Higher is better on every component, so ``sorted(..., reverse=True)`` ranks best-first.
    """
    b = entry.get("brock", {}) or {}
    return (
        1 if b.get("won") else 0,
        1 if (entry.get("reached_pewter") or b.get("reached")) else 0,
        int(entry.get("maps_visited", 0) or 0),
        int(b.get("lead_level") or 0),
        int(entry.get("total_turns", 0) or 0),
    )


def rank_sweep(entries: list[dict]) -> list[dict]:
    """Return entries ranked best-first by prep progress. Pure."""
    return sorted(entries, key=_rank_key, reverse=True)


def build_report(entries: list[dict], n: int, max_turns: int, seed_state: str | None) -> dict:
    """Assemble the sweep leaderboard. Pure."""
    ranked = rank_sweep(entries)
    return {
        "task": "brock_prep_sweep",
        "n": n,
        "max_turns": max_turns,
        "seed_state": seed_state,
        "best": ranked[0] if ranked else None,
        "entries": ranked,
    }


# ---------------------------------------------------------------------------
# Subprocess driver (not unit-tested — exercised by the real run)
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """One config's run: its study, where its artifacts landed, and any captured state."""

    config: SweepConfig
    study: PathStudy
    maps_visited: list[int] = field(default_factory=list)
    captured_state: str | None = None
    returncode: int = 0
    timed_out: bool = False
    rollout_dir: str = ""


def _agent_cmd(
    cfg: Config,
    rom: Path,
    strategy: str,
    fitness_path: Path,
    telemetry_dir: Path,
    pre_brock_state: Path,
    journey_state: Path,
    max_turns: int,
    seed_state: str | None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(cfg.env.agent_script.resolve()),
        str(rom.resolve()),
        "--strategy",
        strategy,
        "--max-turns",
        str(max_turns),
        "--output-json",
        str(fitness_path.resolve()),
        "--telemetry-dir",
        str(telemetry_dir.resolve()),
        "--save-state-on-trainer",
        f"brock:{pre_brock_state.resolve()}",
        "--save-state-every",
        f"500:{journey_state.resolve()}",
        "--config",
        "",  # avoid depending on a local config.toml; JSONL telemetry still writes
    ]
    if seed_state:
        cmd += ["--load-state", str(Path(seed_state).resolve())]
    return cmd


def run_one_config(
    cfg: Config,
    sc: SweepConfig,
    max_turns: int,
    seed_state: str | None,
    sweep_root: Path,
    states_brock_dir: Path,
) -> SweepResult:
    """Run a single prep config in an isolated workspace and study its path."""
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM to run the sweep.")

    rundir = sweep_root / sc.label
    telemetry_dir = rundir / "telemetry"
    fitness_path = rundir / "fitness.json"
    pre_brock_state = rundir / "pre_brock.state"
    journey_state = rundir / "journey.state"
    rom_copy = rundir / "rom.gb"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Per-run ROM copy isolates the .gb.ram shutdown write (the #1 concurrent-run failure mode).
    shutil.copyfile(cfg.env.rom_path, rom_copy)

    env = os.environ.copy()
    env["EVOLVE_PARAMS"] = json.dumps(sc.genome)

    cmd = _agent_cmd(
        cfg,
        rom_copy,
        sc.strategy,
        fitness_path,
        telemetry_dir,
        pre_brock_state,
        journey_state,
        max_turns,
        seed_state,
    )

    returncode, timed_out = 0, False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.env.pokemon_kafka_dir.resolve()),
            env=env,
            capture_output=True,
            timeout=_SWEEP_TIMEOUT_S,
        )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True

    fitness = load_fitness(fitness_path)
    events = load_game_events(telemetry_dir)
    routes = load_routes(cfg.env.routes_json)
    study = build_path_study(events, fitness, routes)

    # Harvest a captured pre-Brock state into the Phase-B ladder dir (named so discover_states
    # picks up the lead level, e.g. ``sweep_explorer_lead_lv12.state``).
    captured = None
    if pre_brock_state.exists():
        states_brock_dir.mkdir(parents=True, exist_ok=True)
        lvl = study.brock.lead_level
        suffix = f"_lead_lv{lvl}" if lvl is not None else ""
        dest = states_brock_dir / f"sweep_{sc.label}{suffix}.state"
        shutil.copyfile(pre_brock_state, dest)
        captured = str(dest)

    return SweepResult(
        config=sc,
        study=study,
        maps_visited=list(extract_visited_maps(events)),
        captured_state=captured,
        returncode=returncode,
        timed_out=timed_out,
        rollout_dir=str(rundir),
    )


def _entry(res: SweepResult) -> dict:
    s = res.study
    return {
        "label": res.config.label,
        "strategy": res.config.strategy,
        "rationale": res.config.rationale,
        "genome_diffs": res.config.diffs(),
        "reached_pewter": s.reached_pewter,
        # Distinct maps reached — the progress proxy. ``extract_visited_maps`` collapses only
        # consecutive dupes, so the raw sequence re-counts a backward-wandering agent; the set
        # does not. The ordered ``map_sequence`` is kept for inspecting the actual path.
        "maps_visited": len(set(res.maps_visited)),
        "map_sequence": res.maps_visited,
        "total_turns": s.total_turns,
        "brock": {
            "reached": s.brock.reached,
            "won": s.brock.won,
            "turns": s.brock.turns,
            "lead_species": s.brock.lead_species,
            "lead_level": s.brock.lead_level,
        },
        "captured_state": res.captured_state,
        "returncode": res.returncode,
        "timed_out": res.timed_out,
        "rollout_dir": res.rollout_dir,
    }


def run_sweep(
    cfg: Config,
    configs: list[SweepConfig],
    max_turns: int,
    seed_state: str | None,
    concurrency: int | None = None,
) -> dict:
    """Run every config concurrently, collect path studies, and write the leaderboard."""
    sweep_root = cfg.storage.out_dir / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    states_brock_dir = Path("./states/brock")
    workers = concurrency or len(configs)

    print(
        f"[sweep] launching {len(configs)} parallel prep agents "
        f"(max_turns={max_turns}, concurrency={workers}, "
        f"seed_state={seed_state or 'intro'})"
    )

    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                run_one_config, cfg, sc, max_turns, seed_state, sweep_root, states_brock_dir
            ): sc
            for sc in configs
        }
        for fut in as_completed(futures):
            sc = futures[fut]
            res = fut.result()
            entry = _entry(res)
            entries.append(entry)
            b = entry["brock"]
            flag = "TIMEOUT " if res.timed_out else ""
            print(
                f"[sweep] {flag}done {sc.label:<15} "
                f"maps={entry['maps_visited']} turns={entry['total_turns']} "
                f"pewter={entry['reached_pewter']} brock={b['reached']} "
                f"won={b['won']} lead_lv={b['lead_level']}"
                f"{' state=' + res.captured_state if res.captured_state else ''}"
            )

    report = build_report(entries, n=len(configs), max_turns=max_turns, seed_state=seed_state)
    out_path = sweep_root / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    _print_leaderboard(report)
    print(f"[sweep] report -> {out_path}")
    captured = [e for e in entries if e["captured_state"]]
    if captured:
        print(
            f"[sweep] captured {len(captured)} pre-Brock state(s) -> states/brock/ (Phase B ready)"
        )
    else:
        print("[sweep] no pre-Brock states captured (the Viridian Forest wall) — see report for "
              "how far each config got.")
    return report


def _print_leaderboard(report: dict) -> None:
    print("\n[sweep] leaderboard (best prep first):")
    header = (
        f"  {'rank':<5}{'label':<16}{'tier':<7}{'maps':<6}{'turns':<7}"
        f"{'pewter':<8}{'brock':<7}{'won':<6}lead_lv"
    )
    print(header)
    for i, e in enumerate(report["entries"], 1):
        b = e["brock"]
        print(
            f"  {i:<5}{e['label']:<16}{e['strategy']:<7}{e['maps_visited']:<6}"
            f"{e['total_turns']:<7}{str(e['reached_pewter']):<8}{str(b['reached']):<7}"
            f"{str(b['won']):<6}{b['lead_level']}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Phase-A prep sweep: N parallel agents learning the road to Brock."
    )
    p.add_argument("--n", type=int, default=10, help="number of parallel prep configs")
    p.add_argument("--max-turns", type=int, default=2500)
    p.add_argument(
        "--seed-state",
        default="./states/route1.state",
        help="save state every run loads to skip the fragile intro (use --from-intro to disable)",
    )
    p.add_argument(
        "--from-intro",
        action="store_true",
        help="start each run from a fresh intro instead of a seed state",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="max simultaneous agents (default: n — run them all at once)",
    )
    args = p.parse_args(argv)

    cfg = load_config()
    seed_state = None if args.from_intro else args.seed_state
    if seed_state and not Path(seed_state).exists():
        print(f"[sweep] seed state {seed_state} not found; falling back to a fresh intro.")
        seed_state = None

    configs = build_sweep(args.n)
    print(f"[sweep] configs: {', '.join(f'{c.label}({c.strategy})' for c in configs)}")
    run_sweep(cfg, configs, args.max_turns, seed_state, concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
