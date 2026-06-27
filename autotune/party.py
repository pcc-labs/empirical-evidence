"""Gen-1 party RAM-poke utility: synthesize pre-Brock save-state variants.

The matchup lever — "which Pokemon at what level beats Brock fastest" — lives in the save
state, not in ``EVOLVE_PARAMS``. This module loads a captured pre-Brock state into a headless
PyBoy, edits the lead Pokemon's RAM, and writes a new state file per matchup cell; the loop
then feeds each via the agent's existing ``--load-state``.

Two layers:
  - a **pure formula layer** (Gen-1 stat math + a curated base-stats table) that is fully
    unit-testable with no emulator, and
  - a thin **RAM I/O layer** over ``pyboy.memory[...]`` (lazy PyBoy import; exercised by the
    smoke run, not unit tests).

Tier 1 (the safe core lever) sets only the lead's **level** and recomputes the dependent
fields (max HP + the four stats), writing only the 44-byte party struct. The captured state is
pre-battle overworld, so the transient battle copy at 0xD0xx is rebuilt from this party data at
battle entry (confirmed by pokemon-kafka/scripts/agent.py reading level from the party struct
"since battle addresses are cleared after battle ends"). No species/DV/move/stat-exp edits, so
validity is preserved by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Party struct layout (offsets within each 44-byte wPartyMon, base = PARTY_BASE + slot*44) ---
PARTY_BASE = 0xD16B
PARTY_STRUCT_SIZE = 44
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES_LIST = 0xD164  # 6 species bytes, mirror of struct offset 0

OFF_SPECIES = 0x00
OFF_CUR_HP = 0x01  # 2 bytes, big-endian
OFF_BOX_LEVEL = 0x03
OFF_STATEXP = 0x11  # HP/Atk/Def/Spd/Spc, 2 bytes each: 0x11,0x13,0x15,0x17,0x19
OFF_DV = 0x1B  # 2 bytes: byte0 = Atk(hi)/Def(lo), byte1 = Spd(hi)/Spc(lo)
OFF_LEVEL = 0x21  # the authoritative level byte
OFF_MAX_HP = 0x22  # 2 bytes, big-endian
OFF_ATK = 0x24
OFF_DEF = 0x26
OFF_SPD = 0x28
OFF_SPC = 0x2A

# Curated Gen-1 base stats (HP, Atk, Def, Speed, Special), keyed by the internal species IDs in
# pokemon-kafka's SPECIES_ID_MAP. pokemon-kafka ships no base-stats dex, so this is the source.
BASE_STATS: dict[int, tuple[int, int, int, int, int]] = {
    0xB0: (39, 52, 43, 65, 50),   # Charmander (the agent's hardcoded starter)
    0xB2: (58, 64, 58, 80, 65),   # Charmeleon
    0xB1: (44, 48, 65, 43, 50),   # Squirtle  (Water — super-effective vs Brock)
    0xB3: (59, 63, 80, 58, 65),   # Wartortle
    0x99: (45, 49, 49, 45, 65),   # Bulbasaur (Grass — super-effective vs Brock)
    0x09: (60, 62, 63, 60, 80),   # Ivysaur
    0x24: (40, 45, 40, 56, 35),   # Pidgey
    0x96: (63, 60, 55, 71, 50),   # Pidgeotto
    0xA5: (30, 56, 35, 72, 25),   # Rattata
    0x54: (35, 55, 30, 90, 50),   # Pikachu
    0x7B: (45, 30, 35, 45, 20),   # Caterpie
    0x6D: (50, 20, 55, 30, 25),   # Metapod
    0x70: (40, 35, 30, 50, 20),   # Weedle
    0x6E: (45, 25, 50, 35, 25),   # Kakuna
}


# ---------------------------------------------------------------------------
# Pure formula layer (no emulator; unit-tested)
# ---------------------------------------------------------------------------


def stat_exp_term(stat_exp: int) -> int:
    """The stat-experience contribution: ``floor(min(ceil(sqrt(stat_exp)), 255) / 4)``."""
    if stat_exp <= 0:
        return 0
    root = math.isqrt(stat_exp - 1) + 1  # exact ceil(sqrt(x)) for x > 0
    return min(root, 255) // 4


def hp_dv(dv_atk: int, dv_def: int, dv_spd: int, dv_spc: int) -> int:
    """Gen-1 HP DV is the concatenation of the four stat DVs' low bits."""
    return ((dv_atk & 1) << 3) | ((dv_def & 1) << 2) | ((dv_spd & 1) << 1) | (dv_spc & 1)


def calc_stat(base: int, dv: int, stat_exp: int, level: int) -> int:
    """Gen-1 non-HP stat at a given level."""
    return ((base + dv) * 2 + stat_exp_term(stat_exp)) * level // 100 + 5


def calc_hp(base_hp: int, hpdv: int, hp_exp: int, level: int) -> int:
    """Gen-1 max-HP at a given level."""
    return ((base_hp + hpdv) * 2 + stat_exp_term(hp_exp)) * level // 100 + level + 10


@dataclass(frozen=True)
class LeadMon:
    """The lead party member's level-independent attributes, decoded from RAM."""

    species: int
    level: int
    dv_atk: int
    dv_def: int
    dv_spd: int
    dv_spc: int
    exp_hp: int
    exp_atk: int
    exp_def: int
    exp_spd: int
    exp_spc: int


def recompute(mon: LeadMon, base: tuple[int, int, int, int, int], level: int) -> dict:
    """Recompute {max_hp, atk, def, spd, spc} for ``mon`` at ``level``. Pure."""
    base_hp, base_atk, base_def, base_spd, base_spc = base
    hpdv = hp_dv(mon.dv_atk, mon.dv_def, mon.dv_spd, mon.dv_spc)
    return {
        "max_hp": calc_hp(base_hp, hpdv, mon.exp_hp, level),
        "atk": calc_stat(base_atk, mon.dv_atk, mon.exp_atk, level),
        "def": calc_stat(base_def, mon.dv_def, mon.exp_def, level),
        "spd": calc_stat(base_spd, mon.dv_spd, mon.exp_spd, level),
        "spc": calc_stat(base_spc, mon.dv_spc, mon.exp_spc, level),
    }


# ---------------------------------------------------------------------------
# RAM I/O layer (thin wrapper over pyboy.memory[...]; not unit-tested)
# ---------------------------------------------------------------------------


def _slot_base(slot: int) -> int:
    return PARTY_BASE + slot * PARTY_STRUCT_SIZE


def _r16(pyboy, addr: int) -> int:
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


def _w16(pyboy, addr: int, value: int) -> None:
    pyboy.memory[addr] = (value >> 8) & 0xFF
    pyboy.memory[addr + 1] = value & 0xFF


def read_lead(pyboy, slot: int = 0) -> LeadMon:
    """Decode the lead party member's level-independent attributes from RAM."""
    b = _slot_base(slot)
    dvs = _r16(pyboy, b + OFF_DV)
    return LeadMon(
        species=pyboy.memory[b + OFF_SPECIES],
        level=pyboy.memory[b + OFF_LEVEL],
        dv_atk=(dvs >> 12) & 0xF,
        dv_def=(dvs >> 8) & 0xF,
        dv_spd=(dvs >> 4) & 0xF,
        dv_spc=dvs & 0xF,
        exp_hp=_r16(pyboy, b + OFF_STATEXP),
        exp_atk=_r16(pyboy, b + OFF_STATEXP + 2),
        exp_def=_r16(pyboy, b + OFF_STATEXP + 4),
        exp_spd=_r16(pyboy, b + OFF_STATEXP + 6),
        exp_spc=_r16(pyboy, b + OFF_STATEXP + 8),
    )


def set_lead_level(pyboy, level: int, slot: int = 0, full_heal: bool = True) -> dict:
    """Set the lead's level and recompute/write HP + the four stats. Returns the read-back."""
    mon = read_lead(pyboy, slot)
    if mon.species not in BASE_STATS:
        raise KeyError(
            f"No base stats for species 0x{mon.species:02X}; add it to BASE_STATS before poking."
        )
    stats = recompute(mon, BASE_STATS[mon.species], level)
    b = _slot_base(slot)
    pyboy.memory[b + OFF_LEVEL] = level
    pyboy.memory[b + OFF_BOX_LEVEL] = level  # keep the box copy consistent
    _w16(pyboy, b + OFF_MAX_HP, stats["max_hp"])
    _w16(pyboy, b + OFF_ATK, stats["atk"])
    _w16(pyboy, b + OFF_DEF, stats["def"])
    _w16(pyboy, b + OFF_SPD, stats["spd"])
    _w16(pyboy, b + OFF_SPC, stats["spc"])
    cur = stats["max_hp"] if full_heal else min(_r16(pyboy, b + OFF_CUR_HP), stats["max_hp"])
    _w16(pyboy, b + OFF_CUR_HP, cur)
    return verify_lead(pyboy, level, slot)


def verify_lead(pyboy, level: int, slot: int = 0) -> dict:
    """Read the lead back and assert level + stats match the formula. Raises on drift."""
    mon = read_lead(pyboy, slot)
    expected = recompute(mon, BASE_STATS[mon.species], level)
    b = _slot_base(slot)
    actual = {
        "level": mon.level,
        "max_hp": _r16(pyboy, b + OFF_MAX_HP),
        "atk": _r16(pyboy, b + OFF_ATK),
        "def": _r16(pyboy, b + OFF_DEF),
        "spd": _r16(pyboy, b + OFF_SPD),
        "spc": _r16(pyboy, b + OFF_SPC),
    }
    if mon.level != level:
        raise AssertionError(f"level not written: got {mon.level}, want {level}")
    for key, want in expected.items():
        if actual[key] != want:
            raise AssertionError(f"{key} drift: wrote {want}, read {actual[key]}")
    return actual


def make_variant(rom_path: str, in_state: str, out_state: str, level: int, slot: int = 0) -> dict:
    """Load ``in_state``, set the lead to ``level``, and save ``out_state``. Emulator-touching."""
    from pyboy import PyBoy

    pyboy = PyBoy(rom_path, window="null")
    try:
        with open(in_state, "rb") as f:
            pyboy.load_state(f)
        result = set_lead_level(pyboy, level, slot=slot)
        with open(out_state, "wb") as f:
            pyboy.save_state(f)
    finally:
        try:
            pyboy.stop()
        except PermissionError:
            pass
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Generate pre-Brock party variants by level.")
    p.add_argument("rom", help="Path to the Pokemon Red ROM")
    p.add_argument("--in-state", required=True, help="Captured pre-Brock save state")
    p.add_argument("--out-dir", default="./states/brock", help="Where to write level variants")
    p.add_argument("--levels", default="10,12,14,16", help="Comma-separated target levels")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in [int(x) for x in args.levels.split(",")]:
        out = out_dir / f"lead_lv{level}.state"
        stats = make_variant(args.rom, args.in_state, str(out), level)
        print(f"[party] L{level}: HP={stats['max_hp']} -> {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
