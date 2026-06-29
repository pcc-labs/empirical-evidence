"""A teacher proposer backed by a local Ollama model (inference-only, no training).

Slots into ``nudge_steer.propose_next_genome(proposer=...)`` exactly like the trained-adapter
proposer: a ``(prompt) -> text`` callable whose text ``parse_genome_response`` turns into a genome.
This lets the loop run end-to-end with a local model *now* — without trained weights and without
the base-model download — by putting an off-the-shelf Ollama model in the seat Claude/the trained
proposer would occupy. It produces behaviour, not distilled weights.

Talks to Ollama's HTTP API via stdlib ``urllib`` (no extra dependency). The pure response handling
(``_extract_content``, ``_strip_think``) is unit-tested; the HTTP call is exercised live.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable

from autotune.config import Config
from autotune.nudge_sft import _SYSTEM

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TIMEOUT_S = 120


def _strip_think(text: str) -> str:
    """Drop any ``<think>...</think>`` block a hybrid-reasoning model emits before the JSON."""
    return _THINK_RE.sub("", text or "").strip()


def _extract_content(response: dict) -> str:
    """Pull the assistant text out of an Ollama ``/api/chat`` response, thinking stripped."""
    return _strip_think((response.get("message") or {}).get("content", ""))


def _chat(cfg: Config, prompt: str) -> str:  # pragma: no cover - live HTTP call
    """POST one chat turn to Ollama and return the (thinking-stripped) assistant text."""
    payload = {
        "model": cfg.ollama.model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,  # hybrid models (qwen3, smollm3): emit the answer, not a reasoning trace
        "options": {"temperature": 0},  # greedy -> deterministic, matching the k=1 contract
    }
    req = urllib.request.Request(
        f"{cfg.ollama.host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return _extract_content(json.loads(resp.read()))


def make_ollama_proposer(cfg: Config) -> Callable[[str], str]:  # pragma: no cover - factory
    """Return a ``(prompt) -> text`` proposer backed by the configured local Ollama model."""
    def _proposer(prompt: str) -> str:
        return _chat(cfg, prompt)

    return _proposer
