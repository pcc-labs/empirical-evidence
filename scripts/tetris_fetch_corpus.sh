#!/usr/bin/env bash
# Pull one corpus out of the private dataset repo onto this box, in the layout
# tetris_corpus --out writes (data/tetris/<corpus_id>/{train,valid,eval}.jsonl), so
# tetris_train_local.sh can read it with CORPUS_DIR=data/tetris.
#
# Needs a valid Hub login (`hf auth login`); the dataset is private.
#
# Usage: scripts/tetris_fetch_corpus.sh <corpus_id>
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS_ID="${1:?usage: scripts/tetris_fetch_corpus.sh <corpus_id>}"
CORPUS_REPO="${CORPUS_REPO:-bdougie/tetris-placements}"
OUT="${CORPUS_ROOT:-data/tetris}"

echo "==> downloading ${CORPUS_REPO}/${CORPUS_ID} into ${OUT}/${CORPUS_ID}"
hf download "$CORPUS_REPO" --repo-type dataset --include "${CORPUS_ID}/*" --local-dir "$OUT"
test -s "${OUT}/${CORPUS_ID}/train.jsonl" || { echo "no train.jsonl landed" >&2; exit 1; }
wc -l "${OUT}/${CORPUS_ID}"/*.jsonl
