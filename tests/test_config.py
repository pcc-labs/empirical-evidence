import pytest

from autotune import config


def test_resolve_mac_profile_by_memory(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_MAC_PROFILE", raising=False)
    assert config.resolve_mac_profile(total_gb=64).name == "m-large"
    assert config.resolve_mac_profile(total_gb=24).name == "m-medium"
    assert config.resolve_mac_profile(total_gb=8).name == "m-lite"
    # Unknown memory falls back to the comfortable default.
    assert config.resolve_mac_profile(total_gb=0).name == "m-large"


def test_resolve_mac_profile_env_override(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_MAC_PROFILE", "m-lite")
    assert config.resolve_mac_profile(total_gb=128).name == "m-lite"


def test_resolve_mac_profile_bad_override(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_MAC_PROFILE", "nope")
    with pytest.raises(ValueError):
        config.resolve_mac_profile()


def test_loop_config_validation():
    assert config.LoopConfig(nudge="both").validate().nudge == "both"
    with pytest.raises(ValueError):
        config.LoopConfig(nudge="rl").validate()
    with pytest.raises(ValueError):
        config.LoopConfig(n_rollouts=0).validate()


def test_with_loop_overrides(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_MAC_PROFILE", raising=False)
    cfg = config.Config()
    updated = cfg.with_loop(generations=7, nudge="sft")
    assert updated.loop.generations == 7
    assert updated.loop.nudge == "sft"
    # Original is untouched (frozen dataclasses).
    assert cfg.loop.generations == config.LoopConfig().generations


def test_storage_derived_paths():
    storage = config.StorageConfig()
    assert storage.sft_dir == storage.data_dir / "sft"
    assert storage.adapter_dir == storage.out_dir / "sft"


def test_resolve_env_and_story(monkeypatch):
    monkeypatch.setenv("POKEMON_KAFKA_DIR", "/tmp/pk")
    monkeypatch.setenv("ROM_PATH", "/tmp/pk/rom/red.gb")
    monkeypatch.setenv("AUTOTUNE_TARGET_MAP", "2")
    env = config.resolve_env()
    assert env.agent_script.as_posix() == "/tmp/pk/scripts/agent.py"
    assert env.routes_json.as_posix() == "/tmp/pk/references/routes.json"
    assert config.resolve_story().target_map_id == 2


def test_resolve_game_default_is_legacy_red(monkeypatch):
    from autotune.config import game_label, resolve_game

    monkeypatch.delenv("AUTOTUNE_GAME", raising=False)
    assert resolve_game() == "red"
    assert game_label() == "Red"


def test_resolve_game_env_override(monkeypatch):
    from autotune.config import game_label, resolve_game

    monkeypatch.setenv("AUTOTUNE_GAME", "yellow")
    assert resolve_game() == "yellow"
    assert game_label() == "Yellow"
    monkeypatch.setenv("AUTOTUNE_GAME", "red_blue")
    assert game_label() == "Red/Blue"


def test_resolve_game_rejects_unknown(monkeypatch):
    import pytest

    from autotune.config import resolve_game

    monkeypatch.setenv("AUTOTUNE_GAME", "crystal")
    with pytest.raises(ValueError):
        resolve_game()
