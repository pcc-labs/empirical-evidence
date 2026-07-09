"""Win-probability harness: learn P(win | situation) from OBSERVED battle outcomes.

The agent emits one ``battle_outcome`` row per fight (pokemon-kafka ``game_events.py``): the
situation at battle start — level gap, my move types, HP buffer, heal item on hand, enemy
species/level/type — paired with the result (won / turns). This module turns a pile of those rows
into a fitted ``P(win | features)`` model.

Two halves:

* **Collection** (``collect_rows`` / ``main``) — an IO/subprocess driver. It sweeps the lead's
  level (``party.make_variant``) over a *neutral* starting state (e.g. Route 2, where Pidgey/Rattata
  are neutral to Charmander's fire) and runs the agent with ``AUTOTUNE_FORCE_FIGHT=1`` so every
  battle resolves to a clean win or faint. Under-levelled runs lose, over-levelled runs win — that
  spread is what makes the probability learnable. Smoke-tested, not unit-tested.

* **Fit** (``featurize`` / ``build_matrix`` / ``fit_winprob`` / ``empirical_winrate``) — pure logic,
  unit-tested. A tiny numpy logistic regression on level gap, HP fraction, heal-on-hand, and a
  one-hot of the enemy's type. The enemy type is a RAW category, never a hand-coded
  "super-effective" flag: if fold-in data spans both fire-weak (forest bug) and neutral (route)
  enemies, the model *discovers* the matchup from the wins, rather than being told it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from autotune.config import load_config
from autotune.genome import base_genome
from autotune.verifier import load_game_events

# Numeric features, in column order. enemy-type one-hot columns are appended after these.
NUMERIC_FEATURES = ("level_gap", "hp_frac", "had_healing")


# --------------------------------------------------------------------------- #
# Pure logic: features + logistic fit (unit-tested)                            #
# --------------------------------------------------------------------------- #


def featurize(row: dict) -> dict:
    """Raw observed features + label for one ``battle_outcome`` row.

    ``hp_frac`` is the start-of-battle HP buffer (0..1). ``enemy_type`` is kept as a raw string
    category — the matchup is learned from outcomes, not encoded here.
    """
    max_hp = max(int(row.get("user_max_hp", 0) or 0), 1)
    hp_start = float(row.get("user_hp_start", 0) or 0)
    return {
        "level_gap": float(row.get("level_gap", 0) or 0),
        "hp_frac": max(0.0, min(1.0, hp_start / max_hp)),
        "had_healing": 1.0 if row.get("had_healing") else 0.0,
        "enemy_type": str(row.get("enemy_type", "") or ""),
        "won": 1 if row.get("won") else 0,
    }


def build_matrix(
    rows: list[dict], enemy_types: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, columns)``: numeric features plus a one-hot of ``enemy_type``.

    ``enemy_types`` pins the one-hot vocabulary (so a fitted model can score new rows with the
    same columns); when ``None`` it is inferred from ``rows``.
    """
    feats = [featurize(r) for r in rows]
    if enemy_types is None:
        enemy_types = sorted({f["enemy_type"] for f in feats if f["enemy_type"]})
    columns = list(NUMERIC_FEATURES) + [f"enemy:{t}" for t in enemy_types]
    X = np.zeros((len(feats), len(columns)), dtype=float)
    y = np.zeros(len(feats), dtype=float)
    type_col = {t: 3 + j for j, t in enumerate(enemy_types)}
    for i, f in enumerate(feats):
        X[i, 0] = f["level_gap"]
        X[i, 1] = f["hp_frac"]
        X[i, 2] = f["had_healing"]
        col = type_col.get(f["enemy_type"])
        if col is not None:
            X[i, col] = 1.0
        y[i] = f["won"]
    return X, y, columns


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


@dataclass
class WinProbModel:
    """A fitted logistic ``P(win)`` model that can score a fresh battle situation."""

    columns: list[str]
    enemy_types: list[str]
    weights: list[float]  # one per column, on standardized features
    bias: float
    mu: list[float]
    sd: list[float]
    n: int = 0
    wins: int = 0

    def predict_proba(self, row: dict) -> float:
        """P(win) for a raw ``battle_outcome``-shaped situation dict."""
        X, _, _ = build_matrix([row], self.enemy_types)
        xs = (X[0] - np.asarray(self.mu)) / np.asarray(self.sd)
        z = float(np.dot(xs, np.asarray(self.weights)) + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "enemy_types": self.enemy_types,
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "mu": [float(m) for m in self.mu],
            "sd": [float(s) for s in self.sd],
            "n": self.n,
            "wins": self.wins,
        }


def fit_winprob(
    rows: list[dict],
    epochs: int = 4000,
    lr: float = 0.2,
    l2: float = 1e-3,
) -> WinProbModel:
    """Fit ``P(win | features)`` with batch-gradient-descent logistic regression (numpy).

    Standardizes features so a single learning rate behaves across columns of different scale.
    Pure and deterministic (zero-initialised), so it is unit-testable without the emulator.
    """
    if not rows:
        raise ValueError("fit_winprob needs at least one battle_outcome row")
    enemy_types = sorted({featurize(r)["enemy_type"] for r in rows if featurize(r)["enemy_type"]})
    X, y, columns = build_matrix(rows, enemy_types)
    Xs, mu, sd = _standardize(X)
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = p - y
        w -= lr * (Xs.T @ grad / n + l2 * w)
        b -= lr * float(grad.mean())
    return WinProbModel(
        columns=columns,
        enemy_types=enemy_types,
        weights=[float(v) for v in w],
        bias=float(b),
        mu=[float(v) for v in mu],
        sd=[float(v) for v in sd],
        n=int(n),
        wins=int(y.sum()),
    )


def empirical_winrate(rows: list[dict], key: str = "level_gap") -> dict:
    """Observed win rate bucketed by a feature (default ``level_gap``).

    Returns ``{bucket: {"n", "wins", "win_rate"}}`` — the ground-truth curve the model is fit to,
    handy for eyeballing whether the data actually has a win/loss spread.
    """
    buckets: dict = {}
    for r in rows:
        b = buckets.setdefault(r.get(key), {"n": 0, "wins": 0})
        b["n"] += 1
        b["wins"] += 1 if r.get("won") else 0
    for b in buckets.values():
        b["win_rate"] = round(b["wins"] / b["n"], 3) if b["n"] else 0.0
    return {k: buckets[k] for k in sorted(buckets, key=lambda x: (x is None, x))}


def summarize(rows: list[dict]) -> dict:
    """High-level dataset stats: counts, overall win rate, and the level-gap curve."""
    wins = sum(1 for r in rows if r.get("won"))
    return {
        "rows": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 3) if rows else 0.0,
        "by_level_gap": empirical_winrate(rows, "level_gap"),
        "enemy_types": sorted({str(r.get("enemy_type", "") or "") for r in rows}),
    }


# --------------------------------------------------------------------------- #
# Collection: level sweep over a neutral state (IO/subprocess, smoke-tested)   #
# --------------------------------------------------------------------------- #


def _battle_outcomes(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if e.get("event_type") == "battle_outcome"]


def collect_rows(
    cfg,
    base_state: str,
    levels: list[int],
    max_turns: int,
    work_root: Path,
    slot: int = 0,
) -> list[dict]:
    """Sweep the lead's level over ``base_state`` and force-fight to harvest win/loss rows.

    For each level: poke a variant state (``party.make_variant``), then run the agent on it with
    ``AUTOTUNE_FORCE_FIGHT=1`` so battles resolve to win/faint. Returns the pooled
    ``battle_outcome`` rows, each tagged with the ``sweep_level`` it was collected at.
    """
    from autotune import party
    from autotune.rollout import run_one

    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Gen-1 ROM (Red/Blue/Yellow).")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    rom = str(cfg.env.rom_path.resolve())

    # run_one does os.environ.copy(), so setting it here propagates force-fight into the agent.
    os.environ["AUTOTUNE_FORCE_FIGHT"] = "1"
    rows: list[dict] = []
    for i, lvl in enumerate(levels):
        variant = work_root / f"lead_lv{lvl}.state"
        try:
            party.make_variant(rom, str(base_state), str(variant), int(lvl), slot)
        except Exception as exc:  # noqa: BLE001 — surface poke failure, keep sweeping
            print(f"[winprob] L{lvl}: party poke failed: {exc}")
            continue
        rollout = run_one(
            cfg,
            base_genome(),
            index=i,
            max_turns=max_turns,
            work_root=work_root,
            load_state=str(variant),
        )
        got = _battle_outcomes(rollout.events)
        for r in got:
            r["sweep_level"] = int(lvl)
        rows.extend(got)
        wins = sum(1 for r in got if r.get("won"))
        print(
            f"[winprob] L{lvl}: {len(got)} battles "
            f"({wins}W {len(got) - wins}L) rc={rollout.returncode}"
        )
    return rows


def _load_extra(telemetry_dir: str) -> list[dict]:
    """Fold in existing ``battle_outcome`` rows from a telemetry dir (e.g. forest fire-vs-bug data),
    so the enemy-type one-hot sees more than one matchup and the model can learn it."""
    return _battle_outcomes(load_game_events(Path(telemetry_dir)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect varied battle outcomes and fit P(win).")
    parser.add_argument("--base-state", default="states/route2.state", help="Neutral state")
    parser.add_argument("--levels", default="5,7,9,11,13,15", help="Comma-separated lead levels")
    parser.add_argument("--max-turns", type=int, default=1500, help="Turns per level run")
    parser.add_argument("--out-dir", default="out/winprob", help="Where to write dataset + model")
    parser.add_argument("--extra-telemetry", default=None, help="Existing telemetry dir to fold in")
    parser.add_argument("--no-collect", action="store_true", help="Fit from dataset.jsonl")
    args = parser.parse_args(argv)

    load_dotenv()  # pick up POKEMON_KAFKA_DIR + ROM_PATH from .env, like the other entrypoints
    cfg = load_config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.jsonl"

    rows: list[dict] = []
    if args.no_collect and dataset_path.is_file():
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    elif not args.no_collect:
        levels = [int(x) for x in args.levels.split(",")]
        rows = collect_rows(cfg, args.base_state, levels, args.max_turns, out_dir / "runs")
    if args.extra_telemetry:
        extra = _load_extra(args.extra_telemetry)
        print(f"[winprob] folding in {len(extra)} rows from {args.extra_telemetry}")
        rows += extra

    if not rows:
        print("[winprob] no battle_outcome rows collected — nothing to fit.")
        return 1

    dataset_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stats = summarize(rows)
    model = fit_winprob(rows)
    (out_dir / "winprob.json").write_text(
        json.dumps({"summary": stats, "model": model.to_dict()}, indent=2)
    )

    print(f"\n[winprob] dataset: {stats['rows']} rows, {stats['wins']}W {stats['losses']}L "
          f"(win_rate={stats['win_rate']})")
    print("[winprob] win rate by level gap:")
    for gap, b in stats["by_level_gap"].items():
        print(f"    gap {gap:>4}: {b['wins']}/{b['n']}  ({b['win_rate']})")
    print(f"[winprob] model -> {out_dir / 'winprob.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
