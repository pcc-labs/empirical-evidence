#!/usr/bin/env bash
# Set autotune up from a sibling pokemon-kafka checkout:
#   1. ensures .gitignore never commits the ROM
#   2. copies the ROM into autotune/rom/
#   3. generates .env (from .env.example) with the correct ROM_PATH + POKEMON_KAFKA_DIR
#
# Idempotent: safe to re-run. Does not overwrite an existing .env — it patches the
# two path lines in place.
#
# Usage: scripts/setup_from_pk.sh [path-to-pokemon-kafka]
#   defaults to $POKEMON_KAFKA_DIR, then ../pokemon-kafka
set -euo pipefail
cd "$(dirname "$0")/.."
AUTOTUNE_ROOT="$(pwd)"

PK="${1:-${POKEMON_KAFKA_DIR:-../pokemon-kafka}}"
PK="$(cd "$PK" 2>/dev/null && pwd || true)"
if [[ -z "$PK" || ! -d "$PK" ]]; then
  echo "error: pokemon-kafka dir not found (tried '${1:-${POKEMON_KAFKA_DIR:-../pokemon-kafka}}')." >&2
  echo "       pass it explicitly: scripts/setup_from_pk.sh /path/to/pokemon-kafka" >&2
  exit 1
fi
echo "==> pokemon-kafka: $PK"

# 1. .gitignore: never commit a ROM (copyrighted) or a local .env.
ensure_ignore() {
  local pattern="$1"
  if ! grep -qxF "$pattern" .gitignore 2>/dev/null; then
    echo "$pattern" >> .gitignore
    echo "    + .gitignore: $pattern"
  fi
}
echo "==> ensuring .gitignore"
touch .gitignore
grep -qF "# ROM (copyrighted) — copied in by setup_from_pk.sh" .gitignore 2>/dev/null \
  || printf '\n# ROM (copyrighted) — copied in by setup_from_pk.sh\n' >> .gitignore
ensure_ignore "rom/"
ensure_ignore "*.gb"
ensure_ignore "*.gb.ram"
ensure_ignore ".env"

# 2. Copy the ROM in.
ROM_SRC="$(ls "$PK"/rom/*.gb 2>/dev/null | head -1 || true)"
if [[ -z "$ROM_SRC" ]]; then
  echo "error: no .gb ROM found in $PK/rom/" >&2
  exit 1
fi
mkdir -p rom
ROM_NAME="$(basename "$ROM_SRC")"
cp -f "$ROM_SRC" "rom/$ROM_NAME"
ROM_DEST="$AUTOTUNE_ROOT/rom/$ROM_NAME"
echo "==> copied ROM -> rom/$ROM_NAME"

# 3. .env: create from example if missing, then patch the path lines.
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> created .env from .env.example"
else
  echo "==> .env exists — patching path lines in place"
fi

# set_kv KEY VALUE — replace existing KEY=... line or append it.
set_kv() {
  local key="$1" val="$2" tmp
  tmp="$(mktemp)"
  if grep -qE "^${key}=" .env; then
    # rewrite the line; use awk to avoid sed delimiter trouble with slashes/spaces.
    awk -v k="$key" -v v="$val" 'BEGIN{FS=OFS="="} $1==k{print k"="v; next} {print}' .env > "$tmp"
    mv "$tmp" .env
  else
    rm -f "$tmp"
    printf '%s=%s\n' "$key" "$val" >> .env
  fi
}
set_kv POKEMON_KAFKA_DIR "$PK"
set_kv ROM_PATH "$ROM_DEST"
echo "    POKEMON_KAFKA_DIR=$PK"
echo "    ROM_PATH=$ROM_DEST"

cat <<EOF

==> done. Next:
    # quick end-to-end (local SFT, no API key):
    scripts/loop.sh 2 2 sft
    # full run, both nudges:
    uv run python -m autotune.loop --nudge both --generations 5 --n 4
    # then apply the winning genome to a real pokemon-kafka run:
    scripts/apply_genome.sh 4000
EOF
