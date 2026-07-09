#!/usr/bin/env bash
# Serve the published forest-lora model (bdougie/smollm3-forest-lora) locally as an
# OpenAI-compatible API via mlx-lm, for pi or anything else that speaks /v1/chat/completions.
#
# Downloads the adapter and fuses it into standalone weights on first run (the fused model
# is what gets served — see docs/serving-forest-lora.md for why --adapter-path is not used).
#
# Usage: scripts/serve_forest_lora.sh [port]      (default port 8080)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="EricFillion/smollm3-3b-mlx"
ADAPTER_REPO="bdougie/smollm3-forest-lora"
ADAPTER_DIR="out/hf/smollm3-forest-lora"
FUSED_DIR="$(pwd)/out/hf/smollm3-forest-fused"
PORT="${1:-8080}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "mlx-lm is Apple-Silicon only; on the CUDA box serve the HF base + adapter with vLLM instead." >&2
  exit 1
fi

if [[ ! -f "$ADAPTER_DIR/adapters.safetensors" ]]; then
  echo "==> downloading $ADAPTER_REPO"
  uv run --extra mac hf download "$ADAPTER_REPO" --local-dir "$ADAPTER_DIR"
fi

if [[ ! -f "$FUSED_DIR/model.safetensors" ]]; then
  echo "==> fusing adapter into $FUSED_DIR"
  uv run --extra mac mlx_lm.fuse \
    --model "$BASE_MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --save-path "$FUSED_DIR"
fi

echo "==> serving $FUSED_DIR at http://127.0.0.1:$PORT/v1"
exec uv run --extra mac mlx_lm.server --model "$FUSED_DIR" --host 127.0.0.1 --port "$PORT"
