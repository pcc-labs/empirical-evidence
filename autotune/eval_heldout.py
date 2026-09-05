"""Held-out eval gate: tuned model must beat base SmolLM3-3B on ground-truth domains.

Scores the two battle domains (battle-outcome: win-field accuracy; move-choice: move/bucket-field
accuracy) and the Forger's two domains (npc-dialogue: body and outcome field accuracy, scored
separately; gate-text: gate field accuracy) on data/sft_v4/valid.jsonl. npc-dialogue's outcome
also gets a majority baseline -- the accuracy of always answering the most common outcome in the
valid rows -- which the tuned score has to beat, since "talk" alone is right most of the time.

``parse_json_answer``/``score_rows``/``majority_baseline``/``gate_passed`` are pure and
unit-tested; the generation loop is a GPU wrapper exercised manually (like train_sft/package).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

BATTLE_DOMAINS = ("battle-outcome", "move-choice")
FORGER_DOMAINS = ("npc-dialogue", "gate-text")
GATED_DOMAINS = BATTLE_DOMAINS + FORGER_DOMAINS

# Metric name -> (domain, answer field). Battle domains score one metric named after the domain
# (first of win/move/bucket present in the expected answer); the Forger domains score named
# fields separately so a body read and an outcome read are reported apart.
FIELD_METRICS: dict[str, tuple[str, str]] = {
    "npc-dialogue/body": ("npc-dialogue", "body"),
    "npc-dialogue/outcome": ("npc-dialogue", "outcome"),
    "gate-text/gate": ("gate-text", "gate"),
}
# Metrics whose tuned score must beat the always-answer-the-mode baseline, not just the base.
BASELINE_METRICS = ("npc-dialogue/outcome",)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_DECODER = json.JSONDecoder()


def parse_json_answer(text: str) -> dict | None:
    """First {...} object in text, or None.

    Decodes from the first brace so nested values (``"items": [...]``, an inner object) survive;
    falls back to the flat-object scan when that brace is not JSON.
    """
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _match(expected: dict, got: dict | None) -> bool:
    if got is None:
        return False
    for key in ("win", "move", "bucket"):
        if key in expected:
            return expected.get(key) == got.get(key)
    return False


def _row_metrics(domain: str, expected: dict, got: dict | None) -> dict[str, bool]:
    """Metric hits for one row: battle domains score one metric, Forger domains one per field."""
    if domain in BATTLE_DOMAINS:
        return {domain: _match(expected, got)}
    return {
        metric: got is not None and expected.get(field) == got.get(field)
        for metric, (d, field) in FIELD_METRICS.items()
        if d == domain
    }


def score_rows(rows: list[dict], predict) -> dict[str, float]:
    """Accuracy per gated metric. ``predict(system, user) -> str``.

    Keys are the domain for battle domains and ``domain/field`` for the Forger domains.
    """
    hits: dict[str, list[bool]] = {}
    for row in rows:
        domain = row.get("domain")
        if domain not in GATED_DOMAINS:
            continue
        system, user = row["messages"][0]["content"], row["messages"][1]["content"]
        expected = json.loads(row["messages"][2]["content"])
        got = parse_json_answer(predict(system, user))
        for metric, hit in _row_metrics(domain, expected, got).items():
            hits.setdefault(metric, []).append(hit)
    return {m: sum(v) / len(v) for m, v in hits.items()}


def row_counts(rows: list[dict]) -> dict[str, int]:
    """Gated rows per domain (what each metric's denominator is)."""
    counts: dict[str, int] = {}
    for row in rows:
        domain = row.get("domain")
        if domain in GATED_DOMAINS:
            counts[domain] = counts.get(domain, 0) + 1
    return counts


def majority_baseline(rows: list[dict], domain: str, field: str) -> dict | None:
    """Accuracy of always answering the most common ``field`` value among ``domain`` rows.

    Returns ``{"label", "accuracy", "n"}`` or None when the domain has no rows. Ties break on
    first appearance.
    """
    counts: dict = {}
    n = 0
    for row in rows:
        if row.get("domain") != domain:
            continue
        value = json.loads(row["messages"][2]["content"]).get(field)
        counts[value] = counts.get(value, 0) + 1
        n += 1
    if not n:
        return None
    label = max(counts, key=counts.__getitem__)
    return {"label": label, "accuracy": counts[label] / n, "n": n}


def majority_baselines(rows: list[dict]) -> dict[str, dict]:
    """``majority_baseline`` for every metric in BASELINE_METRICS that has rows."""
    out = {}
    for metric in BASELINE_METRICS:
        domain, field = FIELD_METRICS[metric]
        baseline = majority_baseline(rows, domain, field)
        if baseline is not None:
            out[metric] = baseline
    return out


def gate_passed(base: dict[str, float], tuned: dict[str, float], baselines: dict) -> bool:
    """Tuned must match or beat base on every scored metric and beat each majority baseline."""
    if any(tuned.get(m, 0) < base.get(m, 0) for m in set(base) | set(tuned)):
        return False
    return all(tuned.get(m, 0) > b["accuracy"] for m, b in baselines.items())


def game_label_of(row: dict) -> str:
    """Which game a row's system prompt names ("Red" for legacy unlabelled prompts)."""
    system = row["messages"][0]["content"]
    if "Pokemon Yellow" in system:
        return "Yellow"
    if "Pokemon Red/Blue" in system:
        return "Red/Blue"
    return "Red"


def score_rows_by_game(rows: list[dict], predict) -> dict[str, dict[str, float]]:
    """Battle-domain accuracy split by the game each prompt names.

    Returns ``{game: {"accuracy": float, "n": count}}``. Lets the eval show the model
    conditioning on Yellow/Blue prompts, not just aggregate Red-dominated numbers. The Forger
    domains are single-game and per-field, so they stay out of this split.
    """
    hits: dict[str, list[bool]] = {}
    for row in rows:
        if row.get("domain") not in BATTLE_DOMAINS:
            continue
        game = game_label_of(row)
        system, user = row["messages"][0]["content"], row["messages"][1]["content"]
        expected = json.loads(row["messages"][2]["content"])
        got = parse_json_answer(predict(system, user))
        hits.setdefault(game, []).append(_match(expected, got))
    return {g: {"accuracy": sum(v) / len(v), "n": len(v)} for g, v in hits.items()}


def _cap_rows_per_domain(rows: list[dict], limit: int) -> list[dict]:
    """Keep at most ``limit`` rows per domain, preserving order."""
    counts: dict[str, int] = {}
    capped: list[dict] = []
    for row in rows:
        domain = row.get("domain")
        if counts.get(domain, 0) >= limit:
            continue
        counts[domain] = counts.get(domain, 0) + 1
        capped.append(row)
    return capped


def _hf_predictor(model_id: str, adapter: str | None):  # pragma: no cover - GPU wrapper
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def predict(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        attention_mask = None
        if torch.is_tensor(enc):
            ids = enc.to(model.device)
        else:
            enc = enc.to(model.device)
            ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

    return predict


def _mlx_predictor(model_path: str, adapter: str | None = None):  # pragma: no cover - MLX wrapper
    """Greedy MLX predictor. Pass ``model_path`` alone (base or fused) or base + ``adapter``.

    ``mlx_lm.load`` applies an adapter correctly (unlike ``mlx_lm.server --adapter-path``); the
    fused model is equivalent and is what production serves.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(model_path, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)  # deterministic

    def predict(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return generate(model, tok, prompt=prompt, max_tokens=80, sampler=sampler, verbose=False)

    return predict


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Held-out eval: tuned vs base.")
    p.add_argument("--valid", type=Path, default=Path("data/sft_v4/valid.jsonl"))
    p.add_argument("--backend", choices=["cuda", "mlx"], default=None, help="default: autodetect")
    p.add_argument("--base", default=None, help="base model id (default: backend base)")
    p.add_argument("--adapter", default="out/sft", help="LoRA adapter dir (cuda/mlx base+adapter)")
    p.add_argument("--model", default=None, help="mlx: eval this full/fused model dir as 'tuned'")
    p.add_argument("--out", type=Path, default=Path("out/eval/heldout.json"))
    p.add_argument("--limit", type=int, default=None, help="cap rows per domain for a quick pass")
    p.add_argument("--by-game", action="store_true", help="also break accuracy down per named game")
    args = p.parse_args(argv)

    from autotune.config import resolve_backend, resolve_model

    backend = args.backend or resolve_backend()
    base = args.base or resolve_model(backend).base_model

    rows = [json.loads(x) for x in args.valid.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = _cap_rows_per_domain(rows, args.limit)

    if backend == "mlx":
        base_predict = _mlx_predictor(base)
        tuned_predict = (
            _mlx_predictor(args.model) if args.model else _mlx_predictor(base, args.adapter)
        )
    else:
        base_predict = _hf_predictor(base, None)
        tuned_predict = _hf_predictor(base, args.adapter)

    t0 = time.monotonic()
    base_scores = score_rows(rows, base_predict)
    t1 = time.monotonic()
    tuned_scores = score_rows(rows, tuned_predict)
    t2 = time.monotonic()
    baselines = majority_baselines(rows)
    passed = gate_passed(base_scores, tuned_scores, baselines)
    result = {
        "backend": backend,
        "adapter": args.model or args.adapter,
        "base": base_scores,
        "tuned": tuned_scores,
        "majority": baselines,
        "rows": row_counts(rows),
        "wall_s": {"base": round(t1 - t0, 1), "tuned": round(t2 - t1, 1)},
        "passed": passed,
    }
    if args.by_game:
        result["base_by_game"] = score_rows_by_game(rows, base_predict)
        result["tuned_by_game"] = score_rows_by_game(rows, tuned_predict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
