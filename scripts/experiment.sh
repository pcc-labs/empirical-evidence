#!/usr/bin/env bash
# Keyless constrained experiment. Captures the scenario's save state if missing, then
# hill-climbs its params from a deliberately-bad start.
#
# Usage: ROM_PATH=... scripts/experiment.sh [nav|battle] [extra args for autotune.experiment]
set -euo pipefail
cd "$(dirname "$0")/.."

SCENARIO="${1:-nav}"
shift || true
PK="${POKEMON_KAFKA_DIR:-../pokemon-kafka}"

if [[ -z "${ROM_PATH:-}" ]]; then
  echo "Set ROM_PATH to your Pokemon Red ROM." >&2
  exit 1
fi
mkdir -p states

case "$SCENARIO" in
  battle)
    STATE_ABS="$(pwd)/states/first_battle.state"
    CAP=(--save-state-on-battle "$STATE_ABS" --battle-limit 1)
    ;;
  nav)
    STATE_ABS="$(pwd)/states/route1.state"
    CAP=(--save-state-on-map "12:$STATE_ABS")
    ;;
  *)
    echo "unknown scenario: $SCENARIO (use nav or battle)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$STATE_ABS" ]]; then
  echo "==> capturing $SCENARIO save state (agent plays until the capture point)..."
  ( cd "$PK" && uv run python scripts/agent.py "$ROM_PATH" "${CAP[@]}" \
      --max-turns 5000 --telemetry-dir "" )
fi
[[ -f "$STATE_ABS" ]] || { echo "capture failed: no $STATE_ABS" >&2; exit 1; }

echo "==> running $SCENARIO experiment"
uv run python -m autotune.experiment --scenario "$SCENARIO" --state "$STATE_ABS" "$@"
