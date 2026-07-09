"""Configuration for autotune.

Mirrors overdub's ``resolve_*`` dataclass pattern. The training/inference backend is selected by
``resolve_backend()``: ``cuda`` (the default on this Linux + RTX 5090 box, via HF/TRL/PEFT) or
``mlx`` (the optional Apple-Silicon path, via ``mlx-lm``). Each backend has its own hardware
profile — ``GPUProfile`` for CUDA, ``MacProfile`` for MLX. Everything is overridable via
``AUTOTUNE_*`` environment variables so the loop can be retargeted without code edits.
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
# Backend selection (cuda | mlx)
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """True if torch is importable and reports a usable CUDA device. Lazy import."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_backend() -> str:
    """Pick the training/inference backend: ``AUTOTUNE_BACKEND`` env > autodetect.

    Autodetect prefers ``cuda`` when a CUDA torch is present, else falls back to ``mlx`` (the
    Apple-Silicon path). An explicit, unknown override fails loudly rather than silently.
    """
    override = os.environ.get("AUTOTUNE_BACKEND")
    if override:
        if override not in {"cuda", "mlx"}:
            raise ValueError(f"Unknown AUTOTUNE_BACKEND={override!r}; choose from ['cuda', 'mlx']")
        return override
    return "cuda" if _cuda_available() else "mlx"


# ---------------------------------------------------------------------------
# CUDA (HF/TRL/PEFT) hardware profile — mirrors overdub's GPUProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUProfile:
    """CUDA training knobs, one profile per VRAM tier. Mirrors overdub's 5090/h100 profiles.

    LoRA is bf16 (no 4-bit) — a 3B base is ~6 GB and the 32 GB card has ample headroom — and the
    genome prompts are short, so ``max_seq_length`` is small and Liger Kernel is unneeded.
    """

    name: str
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


GPU_PROFILES: dict[str, GPUProfile] = {
    # RTX 5090, 32 GB (Blackwell sm_120): comfortable for bf16 3B LoRA.
    "5090": GPUProfile(name="5090"),
    # H100 80 GB (e.g. a brev box): bigger batch, no grad checkpointing.
    "h100": GPUProfile(
        name="h100",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
    ),
}

_DEFAULT_GPU_PROFILE = "5090"


def _autodetect_gpu_profile() -> str | None:
    """Return a ``GPU_PROFILES`` key matching the current device, or None. Lazy torch import."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return None
    return next((key for key in GPU_PROFILES if key in name), None)


def resolve_gpu_profile() -> GPUProfile:
    """Pick a GPU profile: ``AUTOTUNE_GPU_PROFILE`` env > device autodetect > ``5090``."""
    override = os.environ.get("AUTOTUNE_GPU_PROFILE")
    if override:
        if override not in GPU_PROFILES:
            raise ValueError(
                f"Unknown AUTOTUNE_GPU_PROFILE={override!r}; choose from {sorted(GPU_PROFILES)}"
            )
        return GPU_PROFILES[override]
    return GPU_PROFILES[_autodetect_gpu_profile() or _DEFAULT_GPU_PROFILE]


# ---------------------------------------------------------------------------
# Mac (MLX) hardware profile
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


def resolve_profile(backend: str | None = None) -> GPUProfile | MacProfile:
    """Resolve the hardware profile for the active backend (``cuda`` -> GPU, ``mlx`` -> Mac)."""
    if backend is None:
        backend = resolve_backend()
    return resolve_gpu_profile() if backend == "cuda" else resolve_mac_profile()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """The local model the Nudge step trains and queries.

    The default depends on the backend: ``HuggingFaceTB/SmolLM3-3B`` (the HF upstream) on CUDA,
    ``EricFillion/smollm3-3b-mlx`` (the MLX conversion of the same model) on Apple Silicon.
    """

    base_model: str = "HuggingFaceTB/SmolLM3-3B"


# Same SmolLM3-3B, two packagings: HF safetensors for CUDA, MLX conversion for Apple Silicon.
_DEFAULT_BASE_MODEL: dict[str, str] = {
    "cuda": "HuggingFaceTB/SmolLM3-3B",
    "mlx": "EricFillion/smollm3-3b-mlx",
}


def resolve_model(backend: str | None = None) -> ModelConfig:
    """Resolve the base model: ``AUTOTUNE_BASE_MODEL`` env > backend default."""
    if backend is None:
        backend = resolve_backend()
    default = _DEFAULT_BASE_MODEL.get(backend, "HuggingFaceTB/SmolLM3-3B")
    return ModelConfig(base_model=os.environ.get("AUTOTUNE_BASE_MODEL", default))


# ---------------------------------------------------------------------------
# Proposer selection (who proposes the next genome in the Nudge step)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaConfig:
    """A local Ollama model used as the *teacher* proposer (inference-only, no training)."""

    model: str = "qwen3:8b"
    host: str = "http://localhost:11434"


def resolve_ollama() -> OllamaConfig:
    return OllamaConfig(
        model=os.environ.get("AUTOTUNE_OLLAMA_MODEL", "qwen3:8b"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )


# Which game the loop is tuning against, for prompt labels. "red" is the legacy
# default — existing adapters (forest-lora, sft_v3) were trained on "Pokemon Red"
# prompts, so an unset env must not silently shift the prompt distribution.
GAME_LABELS = {"red": "Red", "red_blue": "Red/Blue", "yellow": "Yellow"}


def resolve_game() -> str:
    """``AUTOTUNE_GAME`` env (``red_blue`` | ``yellow``); default legacy ``red``."""
    value = os.environ.get("AUTOTUNE_GAME", "red")
    if value not in GAME_LABELS:
        raise ValueError(f"Unknown AUTOTUNE_GAME={value!r}; choose from {sorted(GAME_LABELS)}")
    return value


def game_label() -> str:
    """Human game name for prompts (e.g. "Red", "Yellow")."""
    return GAME_LABELS[resolve_game()]


def resolve_proposer() -> str:
    """Who proposes the next genome: ``trained`` (the SFT adapter), ``ollama``, or ``heuristic``.

    ``trained`` is the default — use the locally-trained adapter when one exists, else the
    deterministic heuristic. ``ollama`` puts a local Ollama model in the teacher seat so the loop
    runs end-to-end without any trained weights (and without the base-model download).
    """
    value = os.environ.get("AUTOTUNE_PROPOSER", "trained")
    if value not in {"trained", "ollama", "heuristic"}:
        raise ValueError(
            f"Unknown AUTOTUNE_PROPOSER={value!r}; choose from ['trained', 'ollama', 'heuristic']"
        )
    return value


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
    mode: str = "story"  # "story" (route1 per-beat) | "brock" (fewest-turns gym fight)

    def validate(self) -> LoopConfig:
        if self.nudge not in {"sft", "steer", "both"}:
            raise ValueError(f"nudge must be sft|steer|both, got {self.nudge!r}")
        if self.mode not in {"story", "brock", "forest"}:
            raise ValueError(f"mode must be story|brock|forest, got {self.mode!r}")
        if self.n_rollouts < 1:
            raise ValueError("n_rollouts must be >= 1")
        return self


@dataclass(frozen=True)
class Config:
    """Top-level bundle wiring every sub-config together."""

    backend: str = field(default_factory=resolve_backend)
    storage: StorageConfig = field(default_factory=resolve_storage)
    env: EnvConfig = field(default_factory=resolve_env)
    profile: GPUProfile | MacProfile = field(default_factory=resolve_profile)
    model: ModelConfig = field(default_factory=resolve_model)
    proposer: str = field(default_factory=resolve_proposer)
    ollama: OllamaConfig = field(default_factory=resolve_ollama)
    story: StoryConfig = field(default_factory=resolve_story)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def with_loop(self, **kwargs) -> Config:
        """Return a copy with loop fields overridden (CLI flags win over defaults)."""
        return replace(self, loop=replace(self.loop, **kwargs).validate())


def load_config() -> Config:
    """Build the full config from the current environment, wiring profile + model to the backend."""
    backend = resolve_backend()
    return Config(
        backend=backend,
        profile=resolve_profile(backend),
        model=resolve_model(backend),
    )
