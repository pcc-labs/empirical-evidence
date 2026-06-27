#!/usr/bin/env bash
# Apply the best genome autotune found to a real pokemon-kafka run.
# Reads out/best_genome.json, exports its "genome" as EVOLVE_PARAMS, runs the pk agent.
#
# Usage: scripts/apply_genome.sh [max_turns]
set -euo pipefail
cd "$(dirname "$0")/.."

BEST="out/best_genome.json"
PK="${POKEMON_KAFKA_DIR:-../pokemon-kafka}"
MAX_TURNS="${1:-4000}"

if [[ ! -f "$BEST" ]]; then
  echo "No $BEST yet — run the loop first (uv run python -m autotune.loop ...)." >&2
  exit 1
fi
if [[ -z "${ROM_PATH:-}" ]]; then
  echo "Set ROM_PATH to your Pokemon Red ROM." >&2
  exit 1
fi

# Extract just the genome object as compact JSON.
GENOME="$(uv run python -c "import json,sys; print(json.dumps(json.load(open('$BEST'))['genome']))")"
echo "==> applying genome: $GENOME"

cd "$PK"
EVOLVE_PARAMS="$GENOME" uv run python scripts/agent.py "$ROM_PATH" \
  --max-turns "$MAX_TURNS" --strategy medium
