#!/usr/bin/env bash
# Apply the best Brock config empirical-evidence found: load the winning pre-Brock state and play the
# fight with the winning battle genome. Reads out/best_brock.json.
#
# Usage: scripts/apply_brock.sh [max_turns]
set -euo pipefail
cd "$(dirname "$0")/.."

BEST="out/best_brock.json"
PK="${POKEMON_KAFKA_DIR:-../pokemon-kafka}"
MAX_TURNS="${1:-300}"

if [[ ! -f "$BEST" ]]; then
  echo "No $BEST yet — run the brock loop first (uv run python -m autotune.loop --mode brock)." >&2
  exit 1
fi
if [[ -z "${ROM_PATH:-}" ]]; then
  echo "Set ROM_PATH to your Pokemon Red ROM." >&2
  exit 1
fi

GENOME="$(uv run python -c "import json; print(json.dumps(json.load(open('$BEST'))['genome']))")"
STATE="$(uv run python -c "import json; print(json.load(open('$BEST'))['matchup']['state_path'])")"
echo "==> applying Brock genome: $GENOME"
echo "==> from state: $STATE"

cd "$PK"
EVOLVE_PARAMS="$GENOME" uv run python scripts/agent.py "$ROM_PATH" \
  --load-state "$STATE" --battle-limit 1 --max-turns "$MAX_TURNS" --strategy medium
