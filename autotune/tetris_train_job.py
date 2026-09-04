# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.11,<3",
#   "transformers>=5.12,<6",
#   "peft>=0.19,<1",
#   "trl>=1.7,<2",
#   "datasets>=5.0,<6",
#   "accelerate>=1.14,<2",
#   "huggingface_hub",
# ]
# ///
"""LoRA SFT of the Tetris student, with tier-1 eval, in one file.

Runs as `hf jobs uv run autotune/tetris_train_job.py` -- the file is uploaded on
its own, so it imports nothing from this repo -- or locally on the 5090 as
`uv run python -m autotune.tetris_train_job` with CORPUS_DIR set. Its pure parts
(`parse_placement`, `grade_answers`, `resolve_corpus`, `should_upload`) are
unit-tested locally; `main` is the GPU path, smoke-tested.

Env, HF Jobs: CORPUS_REPO (dataset), CORPUS_ID, BASE_MODEL, ADAPTER_REPO (model),
MAX_STEPS (optional), HF_TOKEN (secret).

Env, local: CORPUS_DIR (the `<corpus_id>/` directory tetris_corpus wrote, or its
parent) skips the Hub download and, unless UPLOAD=1, the Hub upload; ADAPTER_DIR
is where the adapter and eval_tier1.json land (default `adapter/`). BATCH_SIZE and
GRAD_ACCUM override the 4 x 4 recipe -- if 32 GB OOMs, drop to 2 x 8 before
reaching for 4-bit.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_placement(text: str) -> tuple[int, int] | None:
    """(rotation, col) from the first {...} candidate in text that parses with
    both keys as ints.

    Scans every brace-delimited candidate rather than taking the first match:
    a non-greedy first-match regex lets a `{` earlier in the text -- a thinking
    preamble, a nested brace -- match a truncated fragment and score a spurious
    parse failure.
    """
    for m in _JSON_RE.finditer(text or ""):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        rot, col = d.get("rotation"), d.get("col")
        if isinstance(rot, bool) or isinstance(col, bool):
            continue
        if not isinstance(rot, int) or not isinstance(col, int):
            continue
        return rot, col
    return None


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


def to_prompt_completion(row: dict) -> dict:
    """`{"messages": [system, user, assistant]}` -> TRL's conversational
    prompt/completion format: `{"prompt": messages[:2], "completion": [messages[2]]}`.

    Fed `messages` rows directly, SFTTrainer computes loss over the whole
    sequence -- the fixed system prompt and the 18-row board, not the ~25-token
    placement. TRL's prompt/completion format masks the prompt tokens from the
    loss instead (see `SFTTrainer`'s dataset-format handling in `trl`), so training
    spends its loss on the placement it is meant to learn. The on-disk corpus
    format (train.jsonl's `messages` rows) is unchanged; this converts at load
    time only, and is pure -- no torch import, safe to call locally.
    """
    messages = row["messages"]
    return {"prompt": messages[:2], "completion": [messages[2]]}


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        sys.exit(f"[train_job] missing env {name}")
    return value


def resolve_corpus(corpus_id: str, corpus_dir: str | None, download) -> Path:
    """The directory holding train.jsonl and eval.jsonl.

    CORPUS_DIR set: a local directory -- the `<corpus_id>/` directory itself, or
    the parent tetris_corpus `--out` and `hf download --local-dir` both write it
    under. No Hub round trip. Unset: `download(corpus_id)` returns the snapshot
    root, as on HF Jobs.
    """
    if corpus_dir:
        root = Path(corpus_dir)
        for candidate in (root / corpus_id, root):
            if (candidate / "train.jsonl").is_file():
                return candidate
        raise FileNotFoundError(
            f"CORPUS_DIR={corpus_dir}: no train.jsonl under {root / corpus_id} or {root}"
        )
    return Path(download(corpus_id)) / corpus_id


def should_upload(env) -> bool:
    """Push the adapter to the Hub? UPLOAD=1/0 decides explicitly. Unset, a local
    corpus (CORPUS_DIR) means a local run and nothing leaves the box."""
    explicit = env.get("UPLOAD")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes"}
    return not env.get("CORPUS_DIR")


def upload_adapter(api, out_dir: Path, adapter_repo: str, corpus_id: str) -> None:
    """Push the trained weights as their own commit, before tier-1 runs at all.

    A tier-1 crash -- OOM at generate, a chat-template edge case, empty
    eval_rows -- must never discard an hour of paid GPU time worth of weights.
    """
    api.create_repo(adapter_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=adapter_repo,
        path_in_repo="adapter",
        commit_message=f"adapter from corpus {corpus_id}",
    )


def generate_tier1(eval_rows: list[dict], generate) -> tuple[dict, list[str]]:
    """Grade a tier-1 answer per held-out row.

    Any exception `generate(row)` raises is caught mid-loop and logged rather
    than propagated, so a grading failure never costs the weights (already
    uploaded by the time this runs) -- it grades whatever answers were produced
    before the failure instead of losing the run.
    """
    answers: list[str] = []
    error: str | None = None
    try:
        for row in eval_rows:
            answers.append(generate(row))
    except Exception as exc:  # noqa: BLE001 — must not cost the run already saved
        error = repr(exc)
        print(f"[train_job] tier-1 generation failed after {len(answers)} rows: {error}")
        traceback.print_exc()
    tier1 = grade_answers(eval_rows[: len(answers)], answers)
    # `{"n": 0, "top1": 0.0}` from a crash is indistinguishable from a genuinely
    # bad student; the error travels with the numbers so the gate can tell.
    tier1["generation_error"] = error
    tier1["eval_rows"] = len(eval_rows)
    return tier1, answers


def upload_tier1(api, out_dir: Path, adapter_repo: str, corpus_id: str, tier1: dict) -> None:
    """Write eval_tier1.json under out_dir and push it as a second commit, then
    tag the corpus -- the tag is the signal that both commits landed."""
    (out_dir / "eval_tier1.json").write_text(json.dumps(tier1, indent=2))
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=adapter_repo,
        path_in_repo="adapter",
        commit_message=f"tier-1 eval for corpus {corpus_id}",
    )
    api.create_tag(adapter_repo, tag=corpus_id, tag_message=f"corpus {corpus_id}", exist_ok=True)


def main() -> int:  # pragma: no cover - GPU path, exercised by the smoke job
    import gc

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    corpus_repo = _env("CORPUS_REPO", "bdougie/tetris-placements")
    corpus_id = _env("CORPUS_ID")
    base_model = _env("BASE_MODEL", "google/gemma-4-E4B-it")
    adapter_repo = _env("ADAPTER_REPO", "bdougie/gemma-4-E4B-tetris-lora")
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    upload = should_upload(os.environ)

    corpus = resolve_corpus(
        corpus_id,
        os.environ.get("CORPUS_DIR"),
        lambda cid: snapshot_download(
            corpus_repo, repo_type="dataset", allow_patterns=[f"{cid}/*"]
        ),
    )
    print(f"[train_job] corpus dir {corpus}; upload={'yes' if upload else 'no'}")
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
        base_model, dtype=torch.bfloat16, device_map="auto"
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

    out_dir = Path(os.environ.get("ADAPTER_DIR", "adapter"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=2,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
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
        # prompt/completion, not the raw `messages` rows: TRL masks the prompt
        # tokens from the loss for this format, so training spends its loss on
        # the ~25-token placement rather than the fixed system prompt and the
        # 18-row board that dwarf it in every row.
        train_dataset=Dataset.from_list([to_prompt_completion(r) for r in train_rows]),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Push the weights now, before tier-1 runs at all: a tier-1 crash must never
    # cost the hour of paid GPU time that produced them. Locally the weights are
    # already on disk under out_dir, so there is nothing to protect.
    api = HfApi() if upload else None
    if upload:
        upload_adapter(api, out_dir, adapter_repo, corpus_id)
        print(f"[train_job] pushed {adapter_repo} adapter for corpus {corpus_id}")
    else:
        print(f"[train_job] adapter saved to {out_dir} (no upload)")

    # Tier 1: generate a placement per held-out board and grade it against the
    # shipped ranking. generate_tier1 catches a mid-loop crash (OOM, a
    # chat-template edge case) and grades whatever was produced before it.
    #
    # Grade the adapter as saved, on a fresh base -- not the in-process model.
    # After SFTTrainer.train() the model's `forward` no longer carries
    # `logits_to_keep` in its signature, so generate() gets [batch, seq, vocab]
    # logits and dies in _sample ("Tensors must have same number of dimensions:
    # got 2 and 3") on the first row: n=0 every run, on any transformers >= 5.
    # Reloading also proves the files on disk are what the package step needs.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto"),
        str(out_dir),
    )
    model.eval()

    def _generate_one(row: dict) -> str:
        ids = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True,
            tokenize=True, return_dict=False, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=64, do_sample=False)
        return tokenizer.decode(gen[0, ids.shape[-1] :], skip_special_tokens=True)

    tier1, answers = generate_tier1(eval_rows, _generate_one)
    tier1["tokens_per_answer"] = (
        sum(len(tokenizer(a)["input_ids"]) for a in answers) / len(answers) if answers else None
    )
    tier1["corpus_id"] = corpus_id
    print("[train_job] tier-1: " + json.dumps(tier1))

    if upload:
        upload_tier1(api, out_dir, adapter_repo, corpus_id, tier1)
        print(f"[train_job] pushed {adapter_repo} tag {corpus_id}")
    else:
        (out_dir / "eval_tier1.json").write_text(json.dumps(tier1, indent=2))
        print(f"[train_job] wrote {out_dir / 'eval_tier1.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
