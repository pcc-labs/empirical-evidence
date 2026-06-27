"""The navigator/battle parameter genome the pokemon-kafka agent accepts via ``EVOLVE_PARAMS``.

Mirrors pokemon-kafka/scripts/evolve.py (``DEFAULT_PARAMS``, ``PARAM_BOUNDS``, ``clamp_params``)
so autotune can build and validate genomes without importing pk's ``scripts/`` package. Keep in
sync if pk's parameter set changes.
"""

from __future__ import annotations

DEFAULT_PARAMS: dict = {
    "stuck_threshold": 8,
    "door_cooldown": 8,
    "waypoint_skip_distance": 3,
    "axis_preference_map_0": "y",
    "bt_max_snapshots": 8,
    "bt_restore_threshold": 15,
    "bt_max_attempts": 3,
    "bt_snapshot_interval": 50,
    "hp_run_threshold": 0.2,
    "hp_heal_threshold": 0.25,
    "unknown_move_score": 10.0,
    "status_move_score": 1.0,
}

# (min, max, type) for numeric params; tuple of allowed values for enums.
PARAM_BOUNDS: dict = {
    "stuck_threshold": (3, 20, int),
    "door_cooldown": (4, 16, int),
    "waypoint_skip_distance": (1, 8, int),
    "axis_preference_map_0": ("x", "y"),
    "bt_max_snapshots": (2, 16, int),
    "bt_restore_threshold": (8, 30, int),
    "bt_max_attempts": (1, 5, int),
    "bt_snapshot_interval": (20, 100, int),
    "hp_run_threshold": (0.05, 0.5, float),
    "hp_heal_threshold": (0.1, 0.6, float),
    "unknown_move_score": (1.0, 30.0, float),
    "status_move_score": (0.0, 10.0, float),
}

# Short descriptions used when prompting an LLM (or the local model) to mutate the genome.
PARAM_DESCRIPTIONS: dict[str, str] = {
    "stuck_threshold": "stuck turns before skipping a waypoint (int, 3-20)",
    "door_cooldown": "frames to walk away from a door after exiting (int, 4-16)",
    "waypoint_skip_distance": "max Manhattan distance to skip a waypoint when stuck (int, 1-8)",
    "axis_preference_map_0": "preferred movement axis on Pallet Town map (x or y)",
    "bt_max_snapshots": "max backtrack snapshots to keep (int, 2-16)",
    "bt_restore_threshold": "stuck turns before restoring a snapshot (int, 8-30)",
    "bt_max_attempts": "max retries from the same snapshot (int, 1-5)",
    "bt_snapshot_interval": "turns between periodic snapshots when not stuck (int, 20-100)",
    "hp_run_threshold": "HP ratio below which to run from wild battles (float, 0.05-0.5)",
    "hp_heal_threshold": "HP ratio below which to use a healing item (float, 0.1-0.6)",
    "unknown_move_score": "baseline score for unknown moves (float, 1.0-30.0)",
    "status_move_score": "score for zero-power status moves (float, 0.0-10.0)",
}


def clamp_params(params: dict) -> dict:
    """Clamp/validate a genome to its bounds, filling invalid fields from defaults. Pure."""
    clamped = dict(params)
    for key, bounds in PARAM_BOUNDS.items():
        if key not in clamped:
            continue
        if all(isinstance(v, str) for v in bounds):  # enum
            if clamped[key] not in bounds:
                clamped[key] = DEFAULT_PARAMS[key]
            continue
        lo, hi, typ = bounds
        try:
            clamped[key] = typ(clamped[key])
        except (ValueError, TypeError):
            clamped[key] = DEFAULT_PARAMS[key]
            continue
        clamped[key] = max(lo, min(hi, clamped[key]))
    return clamped


def base_genome() -> dict:
    """A fresh copy of the default genome."""
    return dict(DEFAULT_PARAMS)
