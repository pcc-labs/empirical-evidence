# Tetris placement distillation, trained on the box

The HF Jobs pipeline in the design spec
(`superpowers/specs/2026-09-04-tetris-placement-distillation-design.md`) runs
unchanged on the RTX 5090. Same scripts, four env knobs, no Hub round trip.

## What has to be true first

| check | how |
|---|---|
| driver loaded, 5090 visible | `nvidia-smi` (the `/verify-nvidia` skill in tetris after a Secure Boot toggle) |
| cu128 torch sees it, bf16 launches on sm_120 | `uv run python smoke_cuda.py` |
| tetris importable for the corpus builder | `uv sync --extra tetris --extra package --group dev` |
| base model on disk | `hf download google/gemma-4-E4B-it` (public, ungated, 16 GB) |
| llama.cpp built once | `git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/code/llama.cpp && cd ~/code/llama.cpp && cmake -B build -G Ninja -DLLAMA_CURL=OFF && cmake --build build --target llama-quantize -j` — `uv tool install cmake ninja` if the box has neither |
| corpus on disk | `scripts/tetris_fetch_corpus.sh <corpus_id>` (private dataset: needs `hf auth login`), or mint one: `uv run python -m autotune.tetris_corpus --runs ../tetris/runs --out data/tetris` |

## The run

```
scripts/tetris_train_local.sh <corpus_id>            # train -> package -> deploy
scripts/tetris_train_local.sh <corpus_id> train      # or one stage at a time
```

- **train**: `autotune.tetris_train_job` with `CORPUS_DIR` (skips
  `snapshot_download`) and `ADAPTER_DIR` (skips `upload_folder` unless
  `UPLOAD=1`). The recipe is the HF Jobs one: bf16 LoRA r=16 on every linear
  projection of the text stack, batch 4 x accumulation 4, gradient checkpointing,
  2 epochs. Tier-1 lands in `out/tetris/<corpus_id>/adapter/eval_tier1.json`.
  If 32 GB OOMs, `BATCH_SIZE=2 GRAD_ACCUM=8` before reaching for 4-bit — the
  first result must not be confounded by quantized training.
- **package**: `autotune.tetris_package_job` with `ADAPTER_DIR`, `OUT_DIR` and
  `LLAMA_CPP_DIR` (skips apt, clone and compile). Merge, `convert_hf_to_gguf.py`
  to f16, `llama-quantize` to Q4_K_M — the quant `gemma4:latest` is served at, so
  the gate measures training, not quantization. The f16 (16 GB) is deleted once
  the quant exists.
- **deploy**: `ollama create gemma4-e4b-tetris:<corpus_id>` from a Modelfile
  naming the GGUF, then `scripts/register_pi_tag.py`. tetris's `pricing.spec`
  lists the whole `pi/gemma4-e4b-tetris:` family with `supports_effort=True`, so
  no per-corpus `pricing.MODELS` edit — an unlisted tag would run effort-free and
  the gate refuses it by name.

The wrapper exports `HF_HUB_OFFLINE=1`: everything is on disk by then, and a
stale `HF_TOKEN` in the shell (it overrides `hf auth login`) cannot 401 a
public-model load.

## Then the gate, as before

```
cd ../tetris && uv run tetris-bench --paused --fixed-effort --no-control --no-power \
  --models pi/gemma4-e4b-tetris:<corpus_id> pi/gemma4:latest \
  --harnesses features --efforts off --seeds 1 2 3 4 5 --max-pieces 60
uv run python -m autotune.tetris_eval --results ../tetris/data/benchmarks/<file>.json \
  --tuned pi/gemma4-e4b-tetris:<corpus_id>
```

## The Hub path still works

Unset `CORPUS_DIR` / `ADAPTER_DIR` and both jobs behave exactly as under
`hf jobs uv run`: download, train, upload, tag. `UPLOAD=1` on a local run
pushes the local adapter and GGUF to the same repos, for a box that wants the
Hub as its archive.

## What was verified on the box (2026-09-04)

Run against an 18-row smoke corpus built from `tests/fixtures/tetris_run`
through the real `sft_row` / `eval_row`, so the prompt bytes were the served
ones:

- **train**: base loads on the 5090, LoRA attaches to the `language_model`
  projections (34.9 M trainable of 7.98 B), a 4 x 4 step runs in ~6 s with
  gradient checkpointing, adapter and `eval_tier1.json` land in `ADAPTER_DIR`.
- **tier-1**: crashed on every row before the reload fix in
  `tetris_train_job.py` — after `SFTTrainer.train()` the model's `forward` loses
  `logits_to_keep` from its signature and generate() dies in `_sample`. That was
  the `{"n": 0, "top1": 0.0}` the handoff warned about; `generation_error` and
  `eval_rows` now travel with the numbers, and the adapter is graded as saved on
  a fresh base.
- **package**: merge on CPU, `convert_hf_to_gguf.py` (llama.cpp's `conversion/`
  package knows `Gemma4ForConditionalGeneration`), `llama-quantize` to Q4_K_M:
  5.3 GB. `ollama create` accepts it (architecture gemma4, 7.5 B) and it answers.

One caveat for the gate: Ollama's `gemma4:latest` is 8.0 B / 9.6 GB at
Q4_K_M — it carries the vision and audio towers and its own quant mix — while
the packaged student is the text stack alone at 5.3 GB. Same nominal quant,
not the same bytes; if the tuned arm loses on quality, rule this out before
blaming the corpus.
