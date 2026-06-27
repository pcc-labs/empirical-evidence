from autotune.genome import DEFAULT_PARAMS, base_genome, clamp_params


def test_base_genome_is_a_copy():
    g = base_genome()
    g["stuck_threshold"] = 99
    assert DEFAULT_PARAMS["stuck_threshold"] != 99


def test_clamp_numeric_bounds():
    clamped = clamp_params({"stuck_threshold": 999, "waypoint_skip_distance": -5})
    assert clamped["stuck_threshold"] == 20  # upper bound
    assert clamped["waypoint_skip_distance"] == 1  # lower bound


def test_clamp_type_coercion():
    clamped = clamp_params({"stuck_threshold": "12", "hp_run_threshold": "0.3"})
    assert clamped["stuck_threshold"] == 12
    assert clamped["hp_run_threshold"] == 0.3


def test_clamp_invalid_value_resets_to_default():
    clamped = clamp_params({"stuck_threshold": "not-a-number"})
    assert clamped["stuck_threshold"] == DEFAULT_PARAMS["stuck_threshold"]


def test_clamp_enum_validation():
    assert clamp_params({"axis_preference_map_0": "z"})["axis_preference_map_0"] == "y"
    assert clamp_params({"axis_preference_map_0": "x"})["axis_preference_map_0"] == "x"


def test_clamp_ignores_unknown_keys():
    clamped = clamp_params({"unknown_key": 5})
    assert clamped["unknown_key"] == 5
