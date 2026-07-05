"""Query the locally-trained model to propose a genome (closes path 1 into the loop).

Two backends behind one ``make_proposer(cfg) -> (prompt) -> text`` contract, where the returned
text is fed to ``nudge_steer.parse_genome_response``:

* ``cuda`` (default) — load the base model + PEFT adapter **once** (cached) and generate in-process
  with greedy decoding. Far faster than a per-call subprocess, which matters because the eval calls
  the proposer repeatedly.
* ``mlx`` — load the base model + trained adapter **once** (cached) via ``mlx_lm`` and generate
  in-process on Apple Silicon. Same rationale as CUDA: eval/benchmark call the proposer repeatedly,
  so a per-call subprocess that reloads the 3B model each time is both slow and (because it passed a
  raw ``--prompt`` with SmolLM3's thinking left on) produced empty, unparseable output.

Train/inference parity is load-bearing: the prompt handed in is the *user* turn from
``nudge_steer.build_mutation_prompt`` — exactly what training paired with the target genome — so we
reconstruct the same ``system``/``user`` chat the SFT examples used (``nudge_sft._SYSTEM``) and
suppress SmolLM3's thinking so the output is the flat JSON the parser expects.

IO/GPU wrapper — exercised by the smoke run, not unit tests.
"""

from __future__ import annotations

import argparse
import gc
import re
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

# Single-entry cache: holds THE most recently used adapter's (model, tokenizer), not one per
# adapter. Repeated calls for the same adapter stay cheap (cache hit); switching adapters evicts
# the resident model and swaps in the new one instead of accumulating one per adapter.
_MLX_CACHE: dict[str, tuple] = {}


def _load_mlx(cfg: Config, adapter_path: Path):  # pragma: no cover - GPU driver
    """Load base model + trained adapter (if present) via mlx_lm, cached by adapter path.

    The cache is single-entry: on a miss, whatever model is currently resident is evicted before
    the new one loads, so switching adapters swaps the resident model rather than accumulating
    one per adapter.
    """
    key = str(Path(adapter_path).resolve())
    if key in _MLX_CACHE:
        return _MLX_CACHE[key]

    from mlx_lm import load

    # Evict the previously cached model (if any) before loading the new one.
    _MLX_CACHE.clear()
    gc.collect()

    if Path(adapter_path).exists():
        model, tokenizer = load(cfg.model.base_model, adapter_path=key)
    else:
        model, tokenizer = load(cfg.model.base_model)
    _MLX_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _generate_mlx(
    cfg: Config, prompt: str, adapter_path: Path, system: str | None = None
) -> str:  # pragma: no cover
    """Greedy-decode one genome from the trained model, mirroring the SFT chat structure.

    Reconstructs the same ``system``/``user`` chat the SFT examples paired (``nudge_sft._SYSTEM`` +
    ``build_mutation_prompt``), disables SmolLM3's thinking, and strips any residual ``<think>``
    block so the output is the flat JSON ``parse_genome_response`` expects. ``system`` overrides
    the default so callers whose examples trained with a different system prompt (e.g. the forest
    benchmark's ``nudge_sft._FOREST_SYSTEM``) don't hit train/inference skew.
    """
    from mlx_lm import generate as mlx_generate

    from autotune.nudge_sft import _SYSTEM

    model, tokenizer = _load_mlx(cfg, adapter_path)
    messages = [
        {"role": "system", "content": system or _SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    out = mlx_generate(model, tokenizer, prompt=text_prompt, max_tokens=_MAX_TOKENS, verbose=False)
    return _THINK_RE.sub("", out).strip()


# ---------------------------------------------------------------------------
# CUDA backend (HF transformers + PEFT)
# ---------------------------------------------------------------------------

# Single-entry cache: holds THE most recently used adapter's (model, tokenizer), not one per
# adapter. Repeated calls for the same adapter stay cheap (cache hit); switching adapters evicts
# the resident model and swaps in the new one instead of accumulating one ~6 GiB bf16 model per
# adapter and OOMing the card.
_CUDA_CACHE: dict[str, tuple] = {}


def _load_cuda(cfg: Config, adapter_path: Path):  # pragma: no cover - GPU driver
    """Load base model + PEFT adapter (if present) onto the GPU, cached by adapter path.

    The cache is single-entry: on a miss, whatever model is currently resident is evicted
    (cleared from the cache, refs dropped, GC + ``torch.cuda.empty_cache()``) before the new
    model loads, so switching adapters swaps the resident model rather than accumulating one
    ~6 GiB bf16 model per adapter and OOMing the card.
    """
    key = str(Path(adapter_path).resolve())
    if key in _CUDA_CACHE:
        return _CUDA_CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Evict the previously cached model (if any) before loading the new one — the cache is
    # single-entry, so we must free the GPU memory first.
    _CUDA_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # Single 3B model on a 32 GB card — load straight to the GPU. device_map="auto" pulls in
    # accelerate's dispatch + quantizer path, which mis-fires under transient memory pressure.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, dtype=torch.bfloat16).to(
        device
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


def _generate_cuda(
    cfg: Config, prompt: str, adapter_path: Path, system: str | None = None
) -> str:  # pragma: no cover
    """Greedy-decode one genome from the trained model, mirroring the SFT chat structure.

    ``system`` overrides the default chat system prompt — see ``_generate_mlx`` for why.
    """
    import torch

    from autotune.nudge_sft import _SYSTEM

    model, tokenizer = _load_cuda(cfg, adapter_path)
    messages = [
        {"role": "system", "content": system or _SYSTEM},
        {"role": "user", "content": prompt},
    ]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=_MAX_TOKENS, do_sample=False)
    text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return _THINK_RE.sub("", text).strip()


def make_proposer(
    cfg: Config, adapter_dir: Path | None = None, system: str | None = None
) -> Callable[[str], str]:
    """Return a ``(prompt) -> text`` proposer backed by the model + adapter (if trained).

    ``adapter_dir`` selects which trained adapter to load; it defaults to
    ``cfg.storage.adapter_dir`` (the final adapter). Pass an explicit dir to benchmark a checkpoint.
    ``system`` overrides the chat system prompt — the forest benchmark passes
    ``nudge_sft._FOREST_SYSTEM`` so inference pairs with what the forest examples trained.
    """
    adapter = adapter_dir if adapter_dir is not None else cfg.storage.adapter_dir

    if cfg.backend == "cuda":
        def _proposer(prompt: str) -> str:
            return _generate_cuda(cfg, prompt, adapter, system)
    else:
        def _proposer(prompt: str) -> str:
            return _generate_mlx(cfg, prompt, adapter, system)

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
