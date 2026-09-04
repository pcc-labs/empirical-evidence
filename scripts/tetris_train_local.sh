#!/usr/bin/env bash
# The HF Jobs pipeline, run in-house on the RTX 5090: train the LoRA, merge +
# convert + quantize to Q4_K_M, `ollama create` the tag, register it with pi.
# Nothing leaves the box; no Hub token is needed once the base model and the
# corpus are on disk (see scripts/tetris_fetch_corpus.sh for the corpus).
#
# Usage: scripts/tetris_train_local.sh <corpus_id> [train|package|deploy|all]
#
#   train    LoRA SFT + tier-1 eval  -> out/tetris/<corpus_id>/adapter/{adapter_*, eval_tier1.json}
#   package  merge, GGUF, Q4_K_M     -> out/tetris/<corpus_id>/gemma-4-E4B-tetris-<corpus_id>-Q4_K_M.gguf
#   deploy   ollama create gemma4-e4b-tetris:<corpus_id> + ~/.pi/agent/models.json
#   all      (default) the three in order
#
# Env knobs: CORPUS_ROOT (data/tetris), OUT_ROOT (out/tetris), LLAMA_CPP_DIR
# (~/code/llama.cpp), BASE_MODEL, MAX_STEPS, BATCH_SIZE / GRAD_ACCUM (4 / 4;
# if 32 GB OOMs use 2 / 8 before reaching for 4-bit), HF_HUB_OFFLINE (1).
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS_ID="${1:?usage: scripts/tetris_train_local.sh <corpus_id> [train|package|deploy|all]}"
STAGE="${2:-all}"
CORPUS_ROOT="${CORPUS_ROOT:-data/tetris}"
OUT_ROOT="${OUT_ROOT:-out/tetris}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/code/llama.cpp}"
OUT_DIR="${OUT_ROOT}/${CORPUS_ID}"
ADAPTER_DIR="${OUT_DIR}/adapter"
TAG="gemma4-e4b-tetris:${CORPUS_ID}"
QUANT="${OUT_DIR}/gemma-4-E4B-tetris-${CORPUS_ID}-Q4_K_M.gguf"
# Everything is on disk by the time this runs; an offline Hub means a stale or
# invalid HF_TOKEN in the shell cannot fail a public-model load with a 401.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

want() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if want train; then
  test -s "${CORPUS_ROOT}/${CORPUS_ID}/train.jsonl" \
    || { echo "no corpus at ${CORPUS_ROOT}/${CORPUS_ID} -- scripts/tetris_fetch_corpus.sh ${CORPUS_ID}, or tetris_corpus --out ${CORPUS_ROOT}" >&2; exit 1; }
  uv run python smoke_cuda.py
  echo "==> train ${CORPUS_ID} -> ${ADAPTER_DIR}"
  CORPUS_ID="$CORPUS_ID" CORPUS_DIR="$CORPUS_ROOT" ADAPTER_DIR="$ADAPTER_DIR" \
    uv run python -m autotune.tetris_train_job
  echo "==> tier-1: $(cat "${ADAPTER_DIR}/eval_tier1.json")"
fi

if want package; then
  test -s "${ADAPTER_DIR}/adapter_config.json" || { echo "no adapter at ${ADAPTER_DIR}" >&2; exit 1; }
  echo "==> package ${ADAPTER_DIR} -> ${QUANT}"
  CORPUS_ID="$CORPUS_ID" ADAPTER_DIR="$ADAPTER_DIR" OUT_DIR="$OUT_DIR" LLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
    uv run python -m autotune.tetris_package_job
fi

if want deploy; then
  test -s "$QUANT" || { echo "no GGUF at ${QUANT}" >&2; exit 1; }
  printf 'FROM %s\n' "$(realpath "$QUANT")" > "${OUT_DIR}/Modelfile"
  echo "==> ollama create ${TAG}"
  ollama create "$TAG" -f "${OUT_DIR}/Modelfile"
  uv run python scripts/register_pi_tag.py "$TAG"
  cat <<MSG

Deployed ${TAG}. tetris's pricing.spec lists the pi/gemma4-e4b-tetris: family
with supports_effort=True, so the gate can run it at effort=off:

  cd ../tetris && uv run tetris-bench --paused --fixed-effort --no-control --no-power \\
    --models pi/${TAG} pi/gemma4:latest --harnesses features --efforts off --seeds 1 2 3 4 5
  uv run python -m autotune.tetris_eval --results ../tetris/data/benchmarks/<file>.json --tuned pi/${TAG}
MSG
fi
