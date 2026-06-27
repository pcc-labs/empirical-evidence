"""Configuration for autotune.

Mirrors overdub's ``resolve_*`` dataclass pattern, but the hardware profile targets
Apple Silicon (MLX) instead of a CUDA GPU. Everything is overridable via ``AUTOTUNE_*``
environment variables so the loop can be retargeted without code edits.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Storage + environment locations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageConfig:
    """Where autotune reads and writes artifacts."""

    data_dir: Path = Path("./data")
    out_dir: Path = Path("./out")
    log_dir: Path = Path("./logs")

    @property
    def sft_dir(self) -> Path:
        return self.data_dir / "sft"

    @property
    def adapter_dir(self) -> Path:
        return self.out_dir / "sft"


def resolve_storage() -> StorageConfig:
    """Resolve storage paths from ``AUTOTUNE_*_DIR`` env vars (falling back to defaults)."""
    return StorageConfig(
        data_dir=Path(os.environ.get("AUTOTUNE_DATA_DIR", "./data")),
        out_dir=Path(os.environ.get("AUTOTUNE_OUT_DIR", "./out")),
        log_dir=Path(os.environ.get("AUTOTUNE_LOG_DIR", "./logs")),
    )


@dataclass(frozen=True)
class EnvConfig:
    """Location of the pokemon-kafka environment autotune drives."""

    pokemon_kafka_dir: Path = Path("../pokemon-kafka")
    rom_path: Path | None = None

    @property
    def agent_script(self) -> Path:
        return self.pokemon_kafka_dir / "scripts" / "agent.py"

    @property
    def routes_json(self) -> Path:
        return self.pokemon_kafka_dir / "references" / "routes.json"


def resolve_env() -> EnvConfig:
    """Resolve pokemon-kafka location + ROM from the environment."""
    pk = Path(os.environ.get("POKEMON_KAFKA_DIR", "../pokemon-kafka")).expanduser()
    rom = os.environ.get("ROM_PATH")
    return EnvConfig(pokemon_kafka_dir=pk, rom_path=Path(rom).expanduser() if rom else None)


# ---------------------------------------------------------------------------
# Mac (MLX) hardware profile — replaces overdub's GPUProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacProfile:
    """Apple-Silicon training + data knobs, sized to unified-memory budget."""

    name: str
    # MLX-LM LoRA knobs
    batch_size: int = 1
    num_layers: int = 16
    max_seq_length: int = 4096
    iters: int = 300
    learning_rate: float = 2e-4
    grad_checkpoint: bool = True
    lora_rank: int = 16
    lora_scale: float = 16.0
    lora_dropout: float = 0.05
    # Data-prep budget (token ceiling per SFT example)
    max_tokens: int = 4096


_PROFILES: dict[str, MacProfile] = {
    # >= 32 GB unified memory (e.g. M4 Max 36 GB): comfortable headroom.
    "m-large": MacProfile(name="m-large", num_layers=16, max_seq_length=4096, iters=300),
    # 16-32 GB: trim sequence length and trainable layers.
    "m-medium": MacProfile(
        name="m-medium", num_layers=8, max_seq_length=2048, iters=200, max_tokens=2048
    ),
    # < 16 GB: smallest footprint that still trains.
    "m-lite": MacProfile(
        name="m-lite", num_layers=4, max_seq_length=1024, iters=120, max_tokens=1024
    ),
}


def _detect_total_gb() -> int:
    """Best-effort unified-memory size in GB via sysctl; 0 if unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip()) // (1024**3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def resolve_mac_profile(total_gb: int | None = None) -> MacProfile:
    """Pick a Mac profile.

    Precedence: ``AUTOTUNE_MAC_PROFILE`` env var > memory-based autodetect > ``m-large``.
    ``total_gb`` is injectable for testing; when ``None`` it is detected via sysctl.
    """
    override = os.environ.get("AUTOTUNE_MAC_PROFILE")
    if override:
        if override not in _PROFILES:
            raise ValueError(
                f"Unknown AUTOTUNE_MAC_PROFILE={override!r}; choose from {sorted(_PROFILES)}"
            )
        return _PROFILES[override]

    gb = _detect_total_gb() if total_gb is None else total_gb
    if gb >= 32:
        return _PROFILES["m-large"]
    if gb >= 16:
        return _PROFILES["m-medium"]
    if gb > 0:
        return _PROFILES["m-lite"]
    return _PROFILES["m-large"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """The local MLX model the Nudge step trains and queries."""

    base_model: str = "EricFillion/smollm3-3b-mlx"


def resolve_model() -> ModelConfig:
    return ModelConfig(
        base_model=os.environ.get("AUTOTUNE_BASE_MODEL", "EricFillion/smollm3-3b-mlx")
    )


# ---------------------------------------------------------------------------
# Story + loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoryConfig:
    """Which story the loop enforces and where it ends."""

    name: str = "route1"
    # Default target = Viridian City (map 1): a fast, reachable demo beat.
    # Set target_map_id=2 to push all the way to Pewter City.
    target_map_id: int = 1


def resolve_story() -> StoryConfig:
    return StoryConfig(
        name=os.environ.get("AUTOTUNE_STORY", "route1"),
        target_map_id=int(os.environ.get("AUTOTUNE_TARGET_MAP", "1")),
    )


@dataclass(frozen=True)
class LoopConfig:
    """One pass = Try -> Check -> Reward -> Nudge, repeated ``generations`` times."""

    n_rollouts: int = 4
    generations: int = 3
    max_turns: int = 1500
    nudge: str = "both"  # "sft" | "steer" | "both"
    concurrency: int = 3
    seed: int = 42

    def validate(self) -> LoopConfig:
        if self.nudge not in {"sft", "steer", "both"}:
            raise ValueError(f"nudge must be sft|steer|both, got {self.nudge!r}")
        if self.n_rollouts < 1:
            raise ValueError("n_rollouts must be >= 1")
        return self


@dataclass(frozen=True)
class Config:
    """Top-level bundle wiring every sub-config together."""

    storage: StorageConfig = field(default_factory=resolve_storage)
    env: EnvConfig = field(default_factory=resolve_env)
    profile: MacProfile = field(default_factory=resolve_mac_profile)
    model: ModelConfig = field(default_factory=resolve_model)
    story: StoryConfig = field(default_factory=resolve_story)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def with_loop(self, **kwargs) -> Config:
        """Return a copy with loop fields overridden (CLI flags win over defaults)."""
        return replace(self, loop=replace(self.loop, **kwargs).validate())


def load_config() -> Config:
    """Build the full config from the current environment."""
    return Config()
