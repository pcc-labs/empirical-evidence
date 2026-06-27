#!/usr/bin/env bash
# One full Try -> Check -> Reward -> Nudge generation, timed per phase.
# Usage: scripts/loop.sh [generations] [n_rollouts] [nudge_mode]
set -euo pipefail
cd "$(dirname "$0")/.."

GENERATIONS="${1:-1}"
N="${2:-3}"
NUDGE="${3:-both}"

echo "==> autotune loop (generations=$GENERATIONS, n=$N, nudge=$NUDGE)"
time uv run python -m autotune.loop \
  --generations "$GENERATIONS" \
  --n "$N" \
  --nudge "$NUDGE"
