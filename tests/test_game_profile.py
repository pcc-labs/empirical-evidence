"""Mirror GameProfile: Yellow is Red/Blue shifted -1 across the 0xCF00-0xD7FF window."""

from autotune.game_profile import RED_BLUE, YELLOW, detect_profile, profile_for_title

SHIFTED_FIELDS = (
    "party_base",
    "addr_party_count",
    "addr_party_species",
    "bag_count_addr",
    "bag_items_addr",
    "addr_battle_hp_hi",
    "addr_battle_max_hp_hi",
)


def test_red_blue_matches_party_module_legacy_constants():
    assert RED_BLUE.name == "red_blue"
    assert RED_BLUE.party_base == 0xD16B
    assert RED_BLUE.addr_party_count == 0xD163
    assert RED_BLUE.addr_party_species == 0xD164
    assert RED_BLUE.bag_count_addr == 0xD31D
    assert RED_BLUE.bag_items_addr == 0xD31E
    assert RED_BLUE.addr_battle_hp_hi == 0xD015
    assert RED_BLUE.addr_battle_max_hp_hi == 0xD023


def test_yellow_shift():
    for f in SHIFTED_FIELDS:
        assert getattr(YELLOW, f) == getattr(RED_BLUE, f) - 1, f
    assert YELLOW.name == "yellow"
    assert YELLOW.label == "Yellow"


def test_title_detection():
    assert profile_for_title("POKEMON YELLOW") is YELLOW
    assert profile_for_title("POKEMON RED") is RED_BLUE
    assert profile_for_title("POKEMON BLUE") is RED_BLUE
    assert profile_for_title(None) is RED_BLUE
    assert profile_for_title("") is RED_BLUE


def test_detect_profile_duck_typed():
    class P:
        cartridge_title = "POKEMON YELLOW"

    assert detect_profile(P()) is YELLOW

    class Q:
        pass

    assert detect_profile(Q()) is RED_BLUE
