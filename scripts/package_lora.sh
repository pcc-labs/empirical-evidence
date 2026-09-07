#!/usr/bin/env bash
# Package a trained LoRA adapter for serving anywhere: merge into the base, convert to GGUF, quantize
# Q4_K_M, upload the GGUF to the adapter's HF repo (gguf/<name>.Q4_K_M.gguf), and register it in the
# local Ollama. Mirrors autotune/tetris_package_job.py for a plain adapter dir.
#
# Usage: scripts/package_lora.sh <adapter_dir> <hf_repo> <ollama_name>
#   e.g. scripts/package_lora.sh out/forger/sft bdougie/smollm3-pokemon-forger-lora pokemon-forger
# Env: LLAMA_CPP_DIR (default ~/code/llama.cpp), BASE_MODEL (default HuggingFaceTB/SmolLM3-3B)
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="$1"; REPO="$2"; NAME="$3"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/code/llama.cpp}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolLM3-3B}"
OUT="out/package/$NAME"; mkdir -p "$OUT"

# A retrained adapter must not ship under a stale merge: if the adapter is newer than the last
# merge (or the GGUFs), throw the old outputs away. Measured 2026-09-07: the v6 Forger was
# uploaded and registered from the morning's v5 GGUF because every file already existed.
if [[ -f "$ADAPTER/adapter_model.safetensors" ]]; then
  for stale in "$OUT/merged/model.safetensors" "$OUT/merged/model.safetensors.index.json" "$OUT/$NAME-f16.gguf" "$OUT/$NAME.Q4_K_M.gguf"; do
    if [[ -f "$stale" && "$ADAPTER/adapter_model.safetensors" -nt "$stale" ]]; then
      echo "adapter is newer than $stale: rebuilding the merge and the GGUFs"
      rm -rf "$OUT/merged" "$OUT/$NAME-f16.gguf" "$OUT/$NAME.Q4_K_M.gguf"
      break
    fi
  done
fi
if [[ ! -f "$OUT/merged/model.safetensors" && ! -f "$OUT/merged/model.safetensors.index.json" ]]; then
  echo "==> merging $ADAPTER into $BASE_MODEL"
  uv run python - "$ADAPTER" "$OUT/merged" "$BASE_MODEL" <<'PY'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
adapter, out, base_id = sys.argv[1:4]
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
merged.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
print("merged ->", out)
PY
fi
F16="$OUT/$NAME-f16.gguf"; QUANT="$OUT/$NAME.Q4_K_M.gguf"
if [[ ! -f "$F16" ]]; then
  echo "==> converting to GGUF (f16)"
  uv run python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$OUT/merged" --outtype f16 --outfile "$F16"
fi
if [[ ! -f "$QUANT" ]]; then
  echo "==> quantizing Q4_K_M"
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$F16" "$QUANT" Q4_K_M
fi
echo "==> uploading $QUANT to $REPO/gguf/"
uv run hf upload "$REPO" "$QUANT" "gguf/$(basename "$QUANT")" --commit-message "Q4_K_M GGUF of the merged adapter, for ollama"
cat > "$OUT/Modelfile" <<MF
FROM ./$(basename "$QUANT")
PARAMETER temperature 0
MF
echo "==> ollama create $NAME:Q4_K_M"
(cd "$OUT" && ollama create "$NAME:Q4_K_M" -f Modelfile)
echo "done: ollama run $NAME:Q4_K_M  |  elsewhere: hf download $REPO gguf/$(basename "$QUANT") && ollama create ..."
