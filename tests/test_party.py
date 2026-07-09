"""Tests for the Gen-1 stat formulas in autotune/party.py.

Anchored against documented Gen-1 values so a wrong formula can't silently produce a
plausible-but-invalid Pokemon. The RAM I/O layer (read_lead/set_lead_level/make_variant) is
exercised by the smoke run, not here.
"""

from __future__ import annotations

from autotune.party import (
    BASE_STATS,
    GROWTH_MEDIUM_FAST,
    GROWTH_MEDIUM_SLOW,
    LeadMon,
    calc_hp,
    calc_stat,
    exp_for_level,
    hp_dv,
    moveset_for_level,
    recompute,
    stat_exp_term,
)

# --- level-up moveset granting ---


def test_moveset_charmander_l30_has_ember_and_recent_moves():
    # The 4 most recently learned by L30: Ember(9), Leer(15), Rage(22), Slash(30). Ember is the
    # move a poked-level Charmander needs to beat Brock — without this it kept only Scratch/Growl.
    ms = moveset_for_level(0xB0, 30)
    assert ms == [(0x34, 25), (0x2B, 30), (0x63, 20), (0xA3, 20)]
    assert 0x34 in [mid for mid, _ in ms]  # Ember present


def test_moveset_charmander_l9_is_scratch_growl_ember():
    assert moveset_for_level(0xB0, 9) == [(0x0A, 35), (0x2D, 40), (0x34, 25)]


def test_moveset_charmander_l1_has_no_ember():
    ms = moveset_for_level(0xB0, 1)
    assert ms == [(0x0A, 35), (0x2D, 40)]
    assert 0x34 not in [mid for mid, _ in ms]


def test_moveset_keeps_only_four_most_recent():
    assert len(moveset_for_level(0xB2, 60)) == 4


def test_moveset_none_for_species_without_learnset():
    assert moveset_for_level(0x24, 5) is None  # Pidgey: leave its moves untouched


def test_exp_for_level_medium_slow_l13():
    # Documented Gen-1 medium-slow total at L13 (Charmander's group); this is the value a poke must
    # write so the level doesn't collapse back after the first battle.
    assert exp_for_level(GROWTH_MEDIUM_SLOW, 13) == 1261


def test_exp_for_level_medium_fast_is_cube():
    assert exp_for_level(GROWTH_MEDIUM_FAST, 13) == 13**3
    assert exp_for_level(GROWTH_MEDIUM_FAST, 10) == 1000


def test_exp_for_level_clamps_and_increases():
    assert exp_for_level(GROWTH_MEDIUM_SLOW, 1) == 0  # formula goes negative at L1 -> clamp
    slow = [exp_for_level(GROWTH_MEDIUM_SLOW, n) for n in range(2, 30)]
    assert slow == sorted(slow) and len(set(slow)) == len(slow)  # strictly increasing


def test_stat_exp_term_boundaries():
    assert stat_exp_term(0) == 0
    assert stat_exp_term(1) == 0
    assert stat_exp_term(16) == 1  # ceil(sqrt(16))=4, //4 = 1
    assert stat_exp_term(65535) == 63  # capped: min(256, 255)//4


def test_hp_dv_concatenates_low_bits():
    assert hp_dv(15, 15, 15, 15) == 15
    assert hp_dv(0, 0, 0, 0) == 0
    assert hp_dv(1, 0, 1, 0) == 0b1010  # atk + spd bits


def test_l100_stat_closed_form():
    # DV 15 + full stat-exp at L100 reduces to 2*base + 98 (Smogon RBY max stat).
    assert calc_stat(130, 15, 65535, 100) == 358  # Gengar Special (base 130)
    assert calc_stat(50, 15, 65535, 100) == 198


def test_l100_hp_closed_form():
    # DV 15 + full stat-exp at L100 reduces to 2*baseHP + 203.
    assert calc_hp(250, 15, 65535, 100) == 703  # Chansey HP (base 250)
    assert calc_hp(44, 15, 65535, 100) == 291


def test_low_level_hand_computed():
    # Zero DV/stat-exp at the levels we actually target vs Brock.
    assert calc_hp(44, 0, 0, 12) == 32  # Squirtle base HP 44
    assert calc_stat(65, 0, 0, 12) == 20  # Squirtle base Def 65


def test_recompute_squirtle_l12():
    mon = LeadMon(
        species=0xB1,
        level=5,
        dv_atk=8,
        dv_def=8,
        dv_spd=8,
        dv_spc=8,
        exp_hp=0,
        exp_atk=0,
        exp_def=0,
        exp_spd=0,
        exp_spc=0,
    )
    stats = recompute(mon, BASE_STATS[0xB1], 12)
    assert stats == {"max_hp": 32, "atk": 18, "def": 22, "spd": 17, "spc": 18}


def test_recompute_scales_with_level():
    mon = LeadMon(
        species=0xB0,
        level=5,
        dv_atk=9,
        dv_def=8,
        dv_spd=10,
        dv_spc=11,
        exp_hp=100,
        exp_atk=0,
        exp_def=0,
        exp_spd=0,
        exp_spc=0,
    )
    low = recompute(mon, BASE_STATS[0xB0], 10)
    high = recompute(mon, BASE_STATS[0xB0], 16)
    assert all(high[k] > low[k] for k in low)  # every stat grows with level


def test_charmander_in_base_stats():
    # The agent's hardcoded starter must be pokeable for the Tier-1 level lever.
    assert 0xB0 in BASE_STATS


def test_recompute_uses_derived_hp_dv():
    # All-odd DVs -> hp_dv 15; flipping to all-even -> hp_dv 0, lowering max HP.
    odd = LeadMon(0xB1, 5, 15, 15, 15, 15, 0, 0, 0, 0, 0)
    even = LeadMon(0xB1, 5, 14, 14, 14, 14, 0, 0, 0, 0, 0)
    odd_hp = recompute(odd, BASE_STATS[0xB1], 14)["max_hp"]
    even_hp = recompute(even, BASE_STATS[0xB1], 14)["max_hp"]
    assert odd_hp > even_hp


# ---------------------------------------------------------------------------
# Game-profile awareness
# ---------------------------------------------------------------------------


class _FakePyBoy:
    """Dict-backed pyboy.memory stand-in (defaults to 0)."""

    class _Mem(dict):
        def __missing__(self, key):
            return 0

    def __init__(self, title="POKEMON RED"):
        self.memory = self._Mem()
        self.cartridge_title = title


def test_slot_base_uses_profile():
    from autotune.game_profile import RED_BLUE, YELLOW
    from autotune.party import PARTY_STRUCT_SIZE, _slot_base

    assert _slot_base(0) == RED_BLUE.party_base  # default stays Red/Blue
    assert _slot_base(1, YELLOW) == YELLOW.party_base + PARTY_STRUCT_SIZE


def test_set_bag_writes_at_yellow_addresses():
    from autotune.game_profile import YELLOW
    from autotune.party import set_bag

    pb = _FakePyBoy(title="POKEMON YELLOW")
    set_bag(pb, [(0x14, 5)], profile=YELLOW)
    assert pb.memory[YELLOW.bag_count_addr] == 1
    assert pb.memory[YELLOW.bag_items_addr] == 0x14
    assert pb.memory[YELLOW.bag_items_addr + 1] == 5
    assert pb.memory[YELLOW.bag_items_addr + 2] == 0xFF


def test_set_bag_detects_profile_from_cartridge():
    from autotune.game_profile import YELLOW
    from autotune.party import set_bag

    pb = _FakePyBoy(title="POKEMON YELLOW")
    set_bag(pb, [(0x14, 5)])  # no profile passed: detect from the pyboy handle
    assert pb.memory[YELLOW.bag_count_addr] == 1


def test_pikachu_moveset_for_level_yellow():
    from autotune.party import MOVE_PP, moveset_for_level

    # L11 Yellow Pikachu: newest 4 of Thundershock/Growl/Tail Whip/T-Wave/Quick Attack
    moves = moveset_for_level(0x54, 11)
    assert moves is not None
    ids = [mid for mid, _pp in moves]
    assert ids == [0x2D, 0x27, 0x56, 0x62]  # Growl, Tail Whip, Thunder Wave, Quick Attack
    for mid, pp in moves:
        assert pp == MOVE_PP[mid]
