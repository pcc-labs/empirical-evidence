"""Held-out eval gate: tuned model must beat base SmolLM3-3B on ground-truth domains.

Scores battle-outcome (win-field accuracy) and move-choice (move/bucket-field accuracy) on
data/sft_v3/valid.jsonl. ``parse_json_answer``/``score_rows`` are pure and unit-tested; the
generation loop is a GPU wrapper exercised manually (like train_sft/package).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GATED_DOMAINS = ("battle-outcome", "move-choice")
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_answer(text: str) -> dict | None:
    """First {...} object in text, or None."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _match(expected: dict, got: dict | None) -> bool:
    if got is None:
        return False
    for key in ("win", "move", "bucket"):
        if key in expected:
            return expected.get(key) == got.get(key)
    return False


def score_rows(rows: list[dict], predict) -> dict[str, float]:
    """Accuracy per gated domain. ``predict(system, user) -> str``."""
    hits: dict[str, list[bool]] = {}
    for row in rows:
        domain = row.get("domain")
        if domain not in GATED_DOMAINS:
            continue
        system, user = row["messages"][0]["content"], row["messages"][1]["content"]
        expected = json.loads(row["messages"][2]["content"])
        got = parse_json_answer(predict(system, user))
        hits.setdefault(domain, []).append(_match(expected, got))
    return {d: sum(v) / len(v) for d, v in hits.items()}


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
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id
            )
        return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

    return predict


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Held-out eval: tuned vs base.")
    p.add_argument("--valid", type=Path, default=Path("data/sft_v3/valid.jsonl"))
    p.add_argument("--base", default="HuggingFaceTB/SmolLM3-3B")
    p.add_argument("--adapter", default="out/sft")
    p.add_argument("--out", type=Path, default=Path("out/eval/heldout.json"))
    p.add_argument("--limit", type=int, default=None, help="cap rows per domain for a quick pass")
    args = p.parse_args(argv)

    rows = [json.loads(x) for x in args.valid.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = _cap_rows_per_domain(rows, args.limit)

    base_scores = score_rows(rows, _hf_predictor(args.base, None))
    tuned_scores = score_rows(rows, _hf_predictor(args.base, args.adapter))
    passed = all(tuned_scores.get(d, 0) > base_scores.get(d, 0) for d in GATED_DOMAINS)
    result = {"base": base_scores, "tuned": tuned_scores, "passed": passed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
