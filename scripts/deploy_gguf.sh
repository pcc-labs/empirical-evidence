#!/usr/bin/env bash
# Pull one corpus's GGUF out of the tetris adapter repo unambiguously and register it
# as an Ollama tag, then as a pi arm.
#
# The adapter repo holds one GGUF per corpus, all under the same repo, named
# gguf/gemma-4-E4B-tetris-<corpus_id>-Q4_K_M.gguf. `ollama pull hf.co/<repo>:Q4_K_M`
# selects by quant tag alone, so from the second corpus onward it resolves *some*
# Q4_K_M file in the repo -- not necessarily this one -- and `ollama cp` labels
# whatever it pulled as the requested corpus regardless. This script names the file
# by corpus id explicitly instead.
#
# Usage: scripts/deploy_gguf.sh <corpus_id>
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS_ID="${1:?usage: scripts/deploy_gguf.sh <corpus_id>}"
ADAPTER_REPO="${ADAPTER_REPO:-bdougie/gemma-4-E4B-tetris-lora}"
FILENAME="gemma-4-E4B-tetris-${CORPUS_ID}-Q4_K_M.gguf"
TAG="gemma4-e4b-tetris:${CORPUS_ID}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> downloading gguf/${FILENAME} from ${ADAPTER_REPO}"
hf download "$ADAPTER_REPO" "gguf/${FILENAME}" --local-dir "$TMP_DIR"

MODELFILE="$TMP_DIR/Modelfile"
echo "FROM ${TMP_DIR}/gguf/${FILENAME}" > "$MODELFILE"

echo "==> ollama create ${TAG}"
ollama create "$TAG" -f "$MODELFILE"

echo "==> registering pi arm for ${TAG}"
uv run python scripts/register_pi_tag.py "$TAG"

cat <<EOF

Deployed ${TAG}. Still needed by hand: add a line to tetris's pricing.MODELS for

    pi/${TAG}

so the tier-2 gate can run effort=off against it (an unlisted tag makes tetris
pricing.spec return supports_effort=False, and the run comes back effort-free).
EOF
