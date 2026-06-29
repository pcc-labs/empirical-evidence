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
    recompute,
    stat_exp_term,
)


def test_exp_for_level_medium_slow_l13():
    # Documented Gen-1 medium-slow total at L13 (Charmander's group); this is the value a poke must
    # write so the level doesn't collapse back after the first battle.
    assert exp_for_level(GROWTH_MEDIUM_SLOW, 13) == 1261


def test_exp_for_level_medium_fast_is_cube():
    assert exp_for_level(GROWTH_MEDIUM_FAST, 13) == 13 ** 3
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
    assert calc_hp(44, 0, 0, 12) == 32   # Squirtle base HP 44
    assert calc_stat(65, 0, 0, 12) == 20  # Squirtle base Def 65


def test_recompute_squirtle_l12():
    mon = LeadMon(
        species=0xB1, level=5,
        dv_atk=8, dv_def=8, dv_spd=8, dv_spc=8,
        exp_hp=0, exp_atk=0, exp_def=0, exp_spd=0, exp_spc=0,
    )
    stats = recompute(mon, BASE_STATS[0xB1], 12)
    assert stats == {"max_hp": 32, "atk": 18, "def": 22, "spd": 17, "spc": 18}


def test_recompute_scales_with_level():
    mon = LeadMon(
        species=0xB0, level=5,
        dv_atk=9, dv_def=8, dv_spd=10, dv_spc=11,
        exp_hp=100, exp_atk=0, exp_def=0, exp_spd=0, exp_spc=0,
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
