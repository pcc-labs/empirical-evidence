# autotune — agent guidelines

## What this is
A local **Try → Check → Reward → Nudge** loop that enforces a *story* in the
[pokemon-kafka](../pokemon-kafka) agent. autotune does not modify pokemon-kafka; it drives
the agent through the already-wired `EVOLVE_PARAMS` env seam and reads its telemetry.

- **Try** — `rollout.py` runs `pokemon-kafka/scripts/agent.py` N times.
- **Check + Reward** — `verifier.py` scores each rollout against an ordered `story.py` spec
  (built from pokemon-kafka's `MAP_PROGRESS` + `routes.json`): per-beat **pass=1 / fail=0**.
- **Nudge** — `nudge_sft.py` (rejection-sampling LoRA SFT of a local model) and/or
  `nudge_steer.py` (mutate the param genome + `notes.md`).
- `loop.py` closes the loop.

## Environment
- **Two backends, selected by `resolve_backend()` (`AUTOTUNE_BACKEND` env > autodetect):**
  - `cuda` (**default here**) — Linux + RTX 5090. Training is bf16 LoRA via HF/TRL/PEFT, base
    `HuggingFaceTB/SmolLM3-3B`. torch comes from the cu128 index (Blackwell sm_120); run
    `uv run python smoke_cuda.py` after `uv sync` to confirm the GPU.
  - `mlx` — Apple Silicon, `mlx-lm` (install with `uv sync --extra mac`), base
    `EricFillion/smollm3-3b-mlx` (the same SmolLM3-3B). Pick the profile to fit unified memory.
- Use **`uv`** for everything: `uv sync`, `uv run python -m autotune.loop ...`, `uv run pytest`.

## Conventions
- Ruff: `E, F, I, W`, line-length 100. `uv run ruff check`.
- Tests: `uv run pytest --cov`. Pure logic (story, verifier, config, SFT-example builder) is
  unit-tested; GPU/subprocess wrappers (`train_sft`, `generate`, `package` drivers) are exercised
  by the smoke run and omitted from coverage.
- Keep pokemon-kafka changes out of scope unless explicitly asked. The `EVOLVE_PARAMS` seam and
  telemetry outputs are the contract.

## Running
```
cp .env.example .env          # set POKEMON_KAFKA_DIR + ROM_PATH
uv sync
uv run python -m autotune.loop --generations 1 --n 3 --nudge both
```
