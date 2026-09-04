# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "trl",
#   "datasets",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
"""LoRA SFT of the Tetris student on HF Jobs, with tier-1 eval, in one file.

Runs as `hf jobs uv run autotune/tetris_train_job.py` -- the file is uploaded on
its own, so it imports nothing from this repo. Its pure parts (`parse_placement`,
`grade_answers`) are unit-tested locally; `main` is the GPU path, smoke-tested.

Env: CORPUS_REPO (dataset), CORPUS_ID, BASE_MODEL, ADAPTER_REPO (model),
MAX_STEPS (optional), HF_TOKEN (secret).

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_placement(text: str) -> tuple[int, int] | None:
    """(rotation, col) from the first {...} in text, or None."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rot, col = d.get("rotation"), d.get("col")
    if isinstance(rot, bool) or isinstance(col, bool):
        return None
    if not isinstance(rot, int) or not isinstance(col, int):
        return None
    return rot, col


def grade_answers(rows: list[dict], answers: list[str]) -> dict:
    """Tier-1 metrics against each row's shipped ranking (best first).

    An unparseable or illegal answer is a parse failure and is excluded from the
    rank and regret means, so those numbers describe the answers that were placements.
    """
    n = len(rows)
    parse_failures = 0
    top1 = top3 = agree = 0
    regrets: list[float] = []
    for row, text in zip(rows, answers):
        choice = parse_placement(text)
        ranking = row["ranking"]
        index = (
            next((i for i, (r, c, _) in enumerate(ranking) if (r, c) == choice), None)
            if choice
            else None
        )
        if index is None:
            parse_failures += 1
            continue
        rank = index + 1
        top1 += rank == 1
        top3 += rank <= 3
        agree += list(choice) == list(row["teacher"])
        best, worst, value = ranking[0][2], ranking[-1][2], ranking[index][2]
        span = best - worst
        regrets.append((best - value) / span if span > 0 else 0.0)
    graded = n - parse_failures
    return {
        "n": n,
        "parse_failures": parse_failures,
        "top1": top1 / n if n else 0.0,
        "top3": top3 / n if n else 0.0,
        "teacher_agreement": agree / n if n else 0.0,
        "mean_regret_norm": sum(regrets) / graded if graded else None,
    }


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        sys.exit(f"[train_job] missing env {name}")
    return value


def main() -> int:  # pragma: no cover - GPU path, exercised by the smoke job
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    corpus_repo = _env("CORPUS_REPO", "bdougie/tetris-placements")
    corpus_id = _env("CORPUS_ID")
    base_model = _env("BASE_MODEL", "google/gemma-4-E4B-it")
    adapter_repo = _env("ADAPTER_REPO", "bdougie/gemma-4-E4B-tetris-lora")
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))

    corpus = (
        Path(snapshot_download(corpus_repo, repo_type="dataset", allow_patterns=[f"{corpus_id}/*"]))
        / corpus_id
    )
    train_lines = (corpus / "train.jsonl").read_text().splitlines()
    eval_lines = (corpus / "eval.jsonl").read_text().splitlines()
    train_rows = [json.loads(x) for x in train_lines if x.strip()]
    eval_rows = [json.loads(x) for x in eval_lines if x.strip()]
    print(
        f"[train_job] corpus {corpus_id}: {len(train_rows)} train rows, {len(eval_rows)} eval rows"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Fail here, not after an hour, if the template cannot take a system turn.
    rendered = tokenizer.apply_chat_template(train_rows[0]["messages"], tokenize=False)
    print("[train_job] rendered example:\n" + rendered[:600])

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    projections = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    try:
        # A multimodal checkpoint keeps its text stack under `language_model`; target only that.
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=rf".*language_model.*\.({projections})$",
            ),
        )
    except ValueError:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=projections.split("|"),
            ),
        )
    model.print_trainable_parameters()

    out_dir = Path("adapter")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=2,
            max_steps=max_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            bf16=True,
            max_length=4096,
            gradient_checkpointing=True,
            logging_steps=10,
            report_to="none",
            save_strategy="no",
        ),
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Tier 1: generate a placement per held-out board and grade it against the shipped ranking.
    model.eval()
    answers: list[str] = []
    for row in eval_rows:
        ids = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=64, do_sample=False)
        answers.append(tokenizer.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True))
    tier1 = grade_answers(eval_rows, answers)
    tier1["tokens_per_answer"] = (
        sum(len(tokenizer(a)["input_ids"]) for a in answers) / len(answers) if answers else None
    )
    tier1["corpus_id"] = corpus_id
    (out_dir / "eval_tier1.json").write_text(json.dumps(tier1, indent=2))
    print("[train_job] tier-1: " + json.dumps(tier1))

    api = HfApi()
    api.create_repo(adapter_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=adapter_repo,
        path_in_repo="adapter",
        commit_message=f"adapter from corpus {corpus_id}",
    )
    api.create_tag(adapter_repo, tag=corpus_id, tag_message=f"corpus {corpus_id}", exist_ok=True)
    print(f"[train_job] pushed {adapter_repo} tag {corpus_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
