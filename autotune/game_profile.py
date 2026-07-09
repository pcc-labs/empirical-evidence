"""Per-game Gen-1 RAM profiles.

Mirror of pokemon-kafka/scripts/game_profile.py (the fields this repo touches) —
keep in sync, per the repo convention of duplicating rather than importing pk code.

Red and Blue (US) share one WRAM layout. Yellow (US) shifts every address in the
0xCF00-0xD7FF block down exactly one byte — verified symbol-by-symbol against the
pret/pokered and pret/pokeyellow disassemblies (wPartyMons d16b→d16a, wNumBagItems
d31d→d31c, wBattleMonHP d015→d014, ...). Map IDs and species internal indices are
identical across all three games, so story specs (MAP_PROGRESS) and species tables
are shared.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

# Yellow's one-byte shift applies to this WRAM window only.
_SHIFT_LO, _SHIFT_HI = 0xCF00, 0xD7FF


@dataclass(frozen=True)
class GameProfile:
    name: str  # matches the "game" field pokemon-kafka stamps into telemetry
    label: str  # human-readable, for prompts/logs

    # Party struct block (see party.py OFF_* offsets within each 44-byte slot)
    party_base: int
    addr_party_count: int
    addr_party_species: int

    # Bag
    bag_count_addr: int
    bag_items_addr: int

    # Battle copies of the lead's HP (capture scripts poke these)
    addr_battle_hp_hi: int  # wBattleMonHP
    addr_battle_max_hp_hi: int  # wBattleMonMaxHP


RED_BLUE = GameProfile(
    name="red_blue",
    label="Red/Blue",
    party_base=0xD16B,
    addr_party_count=0xD163,
    addr_party_species=0xD164,
    bag_count_addr=0xD31D,
    bag_items_addr=0xD31E,
    addr_battle_hp_hi=0xD015,
    addr_battle_max_hp_hi=0xD023,
)


def _shift_wram(profile: GameProfile, delta: int, **overrides) -> GameProfile:
    """Derive a profile by shifting every address inside the WRAM window by ``delta``."""
    changes = {
        f.name: getattr(profile, f.name) + delta
        for f in fields(profile)
        if isinstance(getattr(profile, f.name), int)
        and _SHIFT_LO <= getattr(profile, f.name) <= _SHIFT_HI
    }
    changes.update(overrides)
    return replace(profile, **changes)


YELLOW = _shift_wram(RED_BLUE, -1, name="yellow", label="Yellow")


def profile_for_title(title: str | None) -> GameProfile:
    """Map a cartridge header title to a profile; unknown titles fall back to Red/Blue."""
    t = (title or "").upper()
    if "YELLOW" in t:
        return YELLOW
    return RED_BLUE


def detect_profile(pyboy) -> GameProfile:
    """Auto-detect the game from a loaded PyBoy cartridge's header title."""
    return profile_for_title(getattr(pyboy, "cartridge_title", "") or "")
