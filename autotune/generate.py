"""Query the locally-trained MLX model to propose a genome (closes path 1 into the loop).

Wraps ``mlx_lm generate`` with the trained adapter (when present) and returns the model's text,
which ``nudge_steer.parse_genome_response`` then turns into a genome. Exposes ``make_proposer`` so
the loop can hand the local model to ``nudge_steer.propose_next_genome`` as its ``proposer``.

IO/subprocess wrapper — exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from autotune.config import Config, load_config
from autotune.story import load_story

_MAX_TOKENS = 256


def generate_text(cfg: Config, prompt: str, *, adapter_path: Path | None = None) -> str:
    """Run mlx_lm generate and return the produced text (stdout)."""
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "generate", "--model", cfg.model.base_model]
    if adapter_path is not None and Path(adapter_path).exists():
        cmd += ["--adapter-path", str(Path(adapter_path).resolve())]
    cmd += ["--prompt", prompt, "--max-tokens", str(_MAX_TOKENS)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout


def make_proposer(cfg: Config) -> Callable[[str], str]:
    """Return a ``(prompt) -> text`` proposer backed by the model + adapter (if trained)."""
    adapter = cfg.storage.adapter_dir

    def _proposer(prompt: str) -> str:
        return generate_text(cfg, prompt, adapter_path=adapter)

    return _proposer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the local model to propose a genome.")
    parser.add_argument("--prompt-beat", default="route1", help="story name for the situation")
    args = parser.parse_args(argv)

    cfg = load_config()
    story = load_story(cfg.env.routes_json, args.prompt_beat, cfg.story.target_map_id)
    target = story.target_beat
    prompt = (
        f"Story target: beat {target.beat_id} '{target.name}' (map {target.map_id}). "
        f"Propose the JSON genome that advances furthest along the story."
    )
    print(generate_text(cfg, prompt, adapter_path=cfg.storage.adapter_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
