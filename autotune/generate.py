"""Query the locally-trained model to propose a genome (closes path 1 into the loop).

Two backends behind one ``make_proposer(cfg) -> (prompt) -> text`` contract, where the returned
text is fed to ``nudge_steer.parse_genome_response``:

* ``cuda`` (default) — load the base model + PEFT adapter **once** (cached) and generate in-process
  with greedy decoding. Far faster than a per-call subprocess, which matters because the eval calls
  the proposer repeatedly.
* ``mlx`` — wrap ``mlx_lm generate`` with the trained adapter as a subprocess (Apple Silicon).

Train/inference parity is load-bearing: the prompt handed in is the *user* turn from
``nudge_steer.build_mutation_prompt`` — exactly what training paired with the target genome — so we
reconstruct the same ``system``/``user`` chat the SFT examples used (``nudge_sft._SYSTEM``) and
suppress SmolLM3's thinking so the output is the flat JSON the parser expects.

IO/GPU wrapper — exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from autotune.config import Config, load_config
from autotune.story import load_story

_MAX_TOKENS = 256
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon)
# ---------------------------------------------------------------------------


def generate_text(  # pragma: no cover - subprocess driver
    cfg: Config, prompt: str, *, adapter_path: Path | None = None
) -> str:
    """Run mlx_lm generate and return the produced text (stdout)."""
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "generate", "--model", cfg.model.base_model]
    if adapter_path is not None and Path(adapter_path).exists():
        cmd += ["--adapter-path", str(Path(adapter_path).resolve())]
    cmd += ["--prompt", prompt, "--max-tokens", str(_MAX_TOKENS)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout


# ---------------------------------------------------------------------------
# CUDA backend (HF transformers + PEFT)
# ---------------------------------------------------------------------------

# Cache the loaded (model, tokenizer) per adapter path so eval's repeated calls don't reload.
_CUDA_CACHE: dict[str, tuple] = {}


def _load_cuda(cfg: Config, adapter_path: Path):  # pragma: no cover - GPU driver
    """Load base model + PEFT adapter (if present) onto the GPU once, cached by adapter path."""
    key = str(Path(adapter_path).resolve())
    if key in _CUDA_CACHE:
        return _CUDA_CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok_src = cfg.model.base_model
    if Path(adapter_path).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, key)
        tok_src = key  # adapter dir carries the thinking-disabled chat template
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    _CUDA_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _generate_cuda(cfg: Config, prompt: str, adapter_path: Path) -> str:  # pragma: no cover
    """Greedy-decode one genome from the trained model, mirroring the SFT chat structure."""
    import torch

    from autotune.nudge_sft import _SYSTEM

    model, tokenizer = _load_cuda(cfg, adapter_path)
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=_MAX_TOKENS, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
    return _THINK_RE.sub("", text).strip()


def make_proposer(cfg: Config) -> Callable[[str], str]:
    """Return a ``(prompt) -> text`` proposer backed by the model + adapter (if trained)."""
    adapter = cfg.storage.adapter_dir

    if cfg.backend == "cuda":
        def _proposer(prompt: str) -> str:
            return _generate_cuda(cfg, prompt, adapter)
    else:
        def _proposer(prompt: str) -> str:
            return generate_text(cfg, prompt, adapter_path=adapter)

    return _proposer


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
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
    print(make_proposer(cfg)(prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
