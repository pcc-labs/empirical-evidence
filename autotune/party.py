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

from autotune.game_profile import RED_BLUE, GameProfile, detect_profile

# --- Party struct layout (offsets within each 44-byte wPartyMon, base = party_base + slot*44) ---
# Red/Blue defaults for reference; the RAM I/O layer resolves addresses via a GameProfile
# (auto-detected from the pyboy cartridge title when not passed).
PARTY_BASE = 0xD16B
PARTY_STRUCT_SIZE = 44
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES_LIST = 0xD164  # 6 species bytes, mirror of struct offset 0

OFF_SPECIES = 0x00
OFF_CUR_HP = 0x01  # 2 bytes, big-endian
OFF_BOX_LEVEL = 0x03
OFF_MOVES = 0x08  # 4 move-id bytes
OFF_EXP = 0x0E  # 3 bytes, big-endian — the AUTHORITATIVE experience total
OFF_STATEXP = 0x11  # HP/Atk/Def/Spd/Spc, 2 bytes each: 0x11,0x13,0x15,0x17,0x19
OFF_DV = 0x1B  # 2 bytes: byte0 = Atk(hi)/Def(lo), byte1 = Spd(hi)/Spc(lo)
OFF_PP = 0x1D  # 4 PP bytes (one per move slot)
OFF_LEVEL = 0x21  # the authoritative level byte
OFF_MAX_HP = 0x22  # 2 bytes, big-endian
OFF_ATK = 0x24
OFF_DEF = 0x26
OFF_SPD = 0x28
OFF_SPC = 0x2A

# Gen-1 experience growth group per species (only the two groups our species use). The level byte
# and stats are not authoritative on their own: the game treats the 3-byte EXP total as truth and
# re-derives level (and recomputes stats) from it after the next battle. Poking level/stats WITHOUT
# poking EXP makes a fake level that collapses back on the first fight — observed: an L6 Charmander
# stamped "L13" reverts to L7 mid-forest, fights underleveled, faints, and whites out to Pallet.
GROWTH_MEDIUM_SLOW = "medium_slow"
GROWTH_MEDIUM_FAST = "medium_fast"
GROWTH_GROUP: dict[int, str] = {
    0xB0: GROWTH_MEDIUM_SLOW,
    0xB2: GROWTH_MEDIUM_SLOW,  # Charmander line
    0xB1: GROWTH_MEDIUM_SLOW,
    0xB3: GROWTH_MEDIUM_SLOW,  # Squirtle line
    0x99: GROWTH_MEDIUM_SLOW,
    0x09: GROWTH_MEDIUM_SLOW,  # Bulbasaur line (starters: medium-slow)
    0x24: GROWTH_MEDIUM_FAST,
    0x96: GROWTH_MEDIUM_FAST,  # Pidgey line
    0xA5: GROWTH_MEDIUM_FAST,
    0x54: GROWTH_MEDIUM_FAST,  # Rattata, Pikachu
    0x7B: GROWTH_MEDIUM_FAST,
    0x6D: GROWTH_MEDIUM_FAST,  # Caterpie, Metapod
    0x70: GROWTH_MEDIUM_FAST,
    0x6E: GROWTH_MEDIUM_FAST,  # Weedle, Kakuna
}


def exp_for_level(group: str, level: int) -> int:
    """Minimum EXP total to BE at ``level`` for a growth ``group`` (Gen-1 formulas). Pure."""
    n = level
    if group == GROWTH_MEDIUM_FAST:
        return n**3
    if group == GROWTH_MEDIUM_SLOW:
        return max(0, int(1.2 * n**3 - 15 * n**2 + 100 * n - 140))
    raise KeyError(f"unknown growth group: {group}")


# Curated Gen-1 base stats (HP, Atk, Def, Speed, Special), keyed by the internal species IDs in
# pokemon-kafka's SPECIES_ID_MAP. pokemon-kafka ships no base-stats dex, so this is the source.
BASE_STATS: dict[int, tuple[int, int, int, int, int]] = {
    0xB0: (39, 52, 43, 65, 50),  # Charmander (the agent's hardcoded starter)
    0xB2: (58, 64, 58, 80, 65),  # Charmeleon
    0xB1: (44, 48, 65, 43, 50),  # Squirtle  (Water — super-effective vs Brock)
    0xB3: (59, 63, 80, 58, 65),  # Wartortle
    0x99: (45, 49, 49, 45, 65),  # Bulbasaur (Grass — super-effective vs Brock)
    0x09: (60, 62, 63, 60, 80),  # Ivysaur
    0x24: (40, 45, 40, 56, 35),  # Pidgey
    0x96: (63, 60, 55, 71, 50),  # Pidgeotto
    0xA5: (30, 56, 35, 72, 25),  # Rattata
    0x54: (35, 55, 30, 90, 50),  # Pikachu
    0x7B: (45, 30, 35, 45, 20),  # Caterpie
    0x6D: (50, 20, 55, 30, 25),  # Metapod
    0x70: (40, 35, 30, 50, 20),  # Weedle
    0x6E: (45, 25, 50, 35, 25),  # Kakuna
}


# Gen-1 move id -> base PP, for the moves in LEARNSETS below.
MOVE_PP: dict[int, int] = {
    0x0A: 35,  # Scratch
    0x2D: 40,  # Growl
    0x34: 25,  # Ember
    0x2B: 30,  # Leer
    0x63: 20,  # Rage
    0xA3: 20,  # Slash
    0x35: 25,  # Flamethrower
    0x53: 15,  # Fire Spin
    # Pikachu line (PP from pokeyellow data/moves/moves.asm)
    0x54: 30,  # Thundershock
    0x27: 30,  # Tail Whip
    0x56: 20,  # Thunder Wave
    0x62: 30,  # Quick Attack
    0x68: 15,  # Double Team
    0x15: 20,  # Slam
    0x55: 15,  # Thunderbolt
    0x61: 30,  # Agility
    0x57: 10,  # Thunder
    0x71: 30,  # Light Screen
}

# Gen-1 level-up learnsets (level, move_id), ascending by level. Poking a level WITHOUT the
# matching moves leaves a leveled lead stuck on its capture-time moveset — the reason a poked
# "L30" Charmander kept only Scratch/Growl (no Ember) and could not beat Brock. Only the
# Charmander line is needed: pokemon-kafka hardcodes Charmander, so the pre-Brock lead is this line.
LEARNSETS: dict[int, list[tuple[int, int]]] = {
    0xB0: [  # Charmander
        (1, 0x0A),
        (1, 0x2D),
        (9, 0x34),
        (15, 0x2B),
        (22, 0x63),
        (30, 0xA3),
        (38, 0x35),
        (46, 0x53),
    ],
    0xB2: [  # Charmeleon (evolves at 16)
        (1, 0x0A),
        (1, 0x2D),
        (9, 0x34),
        (15, 0x2B),
        (24, 0x63),
        (33, 0xA3),
        (42, 0x35),
        (56, 0x53),
    ],
    0x54: [  # Pikachu — YELLOW learnset (pokeyellow data/pokemon/evos_moves.asm); the
        # Yellow starter is the use-case for leveling a Pikachu lead. R/B wild Pikachu
        # learn on a different schedule — out of scope until something pokes one.
        (1, 0x54),  # Thundershock
        (1, 0x2D),  # Growl
        (6, 0x27),  # Tail Whip
        (8, 0x56),  # Thunder Wave
        (11, 0x62),  # Quick Attack
        (15, 0x68),  # Double Team
        (20, 0x15),  # Slam
        (26, 0x55),  # Thunderbolt
        (33, 0x61),  # Agility
        (41, 0x57),  # Thunder
        (50, 0x71),  # Light Screen
    ],
}


# ---------------------------------------------------------------------------
# Pure formula layer (no emulator; unit-tested)
# ---------------------------------------------------------------------------


def moveset_for_level(species: int, level: int) -> list[tuple[int, int]] | None:
    """The (move_id, pp) list a species knows at ``level`` — the 4 most recently learned level-up
    moves (Gen-1 keeps the newest 4, oldest pushed out). ``None`` if the species has no learnset
    here (leave its moves untouched). Pure."""
    learnset = LEARNSETS.get(species)
    if learnset is None:
        return None
    known = [(mid, MOVE_PP[mid]) for lv, mid in learnset if lv <= level]
    return known[-4:]


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


def _slot_base(slot: int, profile: GameProfile = RED_BLUE) -> int:
    return profile.party_base + slot * PARTY_STRUCT_SIZE


def _r16(pyboy, addr: int) -> int:
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


def _w16(pyboy, addr: int, value: int) -> None:
    pyboy.memory[addr] = (value >> 8) & 0xFF
    pyboy.memory[addr + 1] = value & 0xFF


def _r24(pyboy, addr: int) -> int:
    return (pyboy.memory[addr] << 16) | (pyboy.memory[addr + 1] << 8) | pyboy.memory[addr + 2]


def _w24(pyboy, addr: int, value: int) -> None:
    pyboy.memory[addr] = (value >> 16) & 0xFF
    pyboy.memory[addr + 1] = (value >> 8) & 0xFF
    pyboy.memory[addr + 2] = value & 0xFF


def read_lead(pyboy, slot: int = 0, profile: GameProfile | None = None) -> LeadMon:
    """Decode the lead party member's level-independent attributes from RAM."""
    profile = profile or detect_profile(pyboy)
    b = _slot_base(slot, profile)
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


def set_lead_level(
    pyboy,
    level: int,
    slot: int = 0,
    full_heal: bool = True,
    grant_moves: bool = True,
    profile: GameProfile | None = None,
) -> dict:
    """Set the lead's level and recompute/write HP + the four stats. With ``grant_moves`` (default),
    also grant the level-appropriate moveset (so a poked lead isn't stuck on its capture-time
    moves — e.g. a leveled Charmander gets Ember, which it needs to beat Brock). Returns the
    read-back."""
    profile = profile or detect_profile(pyboy)
    mon = read_lead(pyboy, slot, profile)
    if mon.species not in BASE_STATS:
        raise KeyError(
            f"No base stats for species 0x{mon.species:02X}; add it to BASE_STATS before poking."
        )
    if mon.species not in GROWTH_GROUP:
        raise KeyError(
            f"No growth group for 0x{mon.species:02X}; add it to GROWTH_GROUP before poking."
        )
    stats = recompute(mon, BASE_STATS[mon.species], level)
    b = _slot_base(slot, profile)
    pyboy.memory[b + OFF_LEVEL] = level
    pyboy.memory[b + OFF_BOX_LEVEL] = level  # keep the box copy consistent
    # Authoritative: set EXP to the level's threshold, or the game re-derives the level from stale
    # EXP after the next battle and recomputes stats down to the real level.
    _w24(pyboy, b + OFF_EXP, exp_for_level(GROWTH_GROUP[mon.species], level))
    _w16(pyboy, b + OFF_MAX_HP, stats["max_hp"])
    _w16(pyboy, b + OFF_ATK, stats["atk"])
    _w16(pyboy, b + OFF_DEF, stats["def"])
    _w16(pyboy, b + OFF_SPD, stats["spd"])
    _w16(pyboy, b + OFF_SPC, stats["spc"])
    cur = stats["max_hp"] if full_heal else min(_r16(pyboy, b + OFF_CUR_HP), stats["max_hp"])
    _w16(pyboy, b + OFF_CUR_HP, cur)
    if grant_moves:
        moves = moveset_for_level(mon.species, level)
        if moves is not None:  # only species with a known learnset; else leave moves untouched
            for i in range(4):
                mid, pp = moves[i] if i < len(moves) else (0, 0)
                pyboy.memory[b + OFF_MOVES + i] = mid
                pyboy.memory[b + OFF_PP + i] = pp
    return verify_lead(pyboy, level, slot, profile)


def verify_lead(pyboy, level: int, slot: int = 0, profile: GameProfile | None = None) -> dict:
    """Read the lead back and assert level + stats match the formula. Raises on drift."""
    profile = profile or detect_profile(pyboy)
    mon = read_lead(pyboy, slot, profile)
    expected = recompute(mon, BASE_STATS[mon.species], level)
    b = _slot_base(slot, profile)
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
    exp = _r24(pyboy, b + OFF_EXP)
    want_exp = exp_for_level(GROWTH_GROUP[mon.species], level)
    if exp != want_exp:
        raise AssertionError(f"exp not written: got {exp}, want {want_exp} (level would collapse)")
    for key, want in expected.items():
        if actual[key] != want:
            raise AssertionError(f"{key} drift: wrote {want}, read {actual[key]}")
    return actual


# Red/Blue defaults for reference; set_bag resolves via the game profile.
BAG_COUNT_ADDR = 0xD31D
BAG_ITEMS_ADDR = 0xD31E  # pairs of [item_id, quantity], 0xFF-terminated (Gen-1)
# Gen-1 item ids (decimal): 20=Potion, 19=Super Potion, 18=Hyper Potion. NOTE pokemon-kafka's
# HEALING_ITEM_IDS mislabels 0x19 (25 = SoulBadge) as "Super Potion" — do NOT use 0x19 here.
POTION_ID = 0x14  # 20 = Potion (heals 20 HP) — confirmed correct
SUPER_POTION_ID = 0x13  # 19 = Super Potion (heals 50 HP)


def set_bag(pyboy, items: list[tuple[int, int]], profile: GameProfile | None = None) -> None:
    """Overwrite the bag with ``(item_id, quantity)`` pairs, 0xFF-terminated. Emulator-touching.

    The forest states ship with an empty bag, so the agent's ``hp_heal_threshold`` is inert and the
    lead bleeds out in the grass. Stocking potions makes healing a live survival lever.
    """
    profile = profile or detect_profile(pyboy)
    pyboy.memory[profile.bag_count_addr] = len(items)
    addr = profile.bag_items_addr
    for item_id, qty in items:
        pyboy.memory[addr] = item_id & 0xFF
        pyboy.memory[addr + 1] = qty & 0xFF
        addr += 2
    pyboy.memory[addr] = 0xFF  # terminator


def stock_potions(rom_path: str, in_state: str, out_state: str, potions: int = 30) -> None:
    """Load ``in_state``, stock the bag with Potions, save ``out_state``. Emulator-side.

    Stocks plain Potions (0x14): it's the only healing id pokemon-kafka's ``find_healing_item``
    actually recognizes, so the agent can detect and the follower can apply them.
    """
    from pyboy import PyBoy

    pyboy = PyBoy(rom_path, window="null")
    try:
        with open(in_state, "rb") as f:
            pyboy.load_state(f)
        set_bag(pyboy, [(POTION_ID, potions)])
        with open(out_state, "wb") as f:
            pyboy.save_state(f)
    finally:
        try:
            pyboy.stop()
        except PermissionError:
            pass


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
    p.add_argument("rom", help="Path to a Gen-1 Pokemon ROM (Red/Blue/Yellow)")
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
