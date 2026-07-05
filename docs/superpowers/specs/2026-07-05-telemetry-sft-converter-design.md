# Telemetry → SFT converter design

**Date:** 2026-07-05
**Goal:** Convert `pokemon.game.v1` telemetry (both repos) into a multi-domain SFT corpus large
enough that retraining SmolLM3-3B produces weights worth publishing to Hugging Face Hub — replacing
the current 11-example memorized corpus.

## Decisions (from brainstorming)

- **Model task:** multi-task blend — battle advisor + genome proposer + narrator, one model.
- **Hosting target:** Hugging Face Hub (fused checkpoint + LoRA adapter + model card + manifest).
- **Training scope for v1:** SFT only. DPO deferred.
- **Training hardware:** this machine (RTX 5090), existing TRL/PEFT path.
- **Approach:** deterministic pure-Python converter (Approach A). No Kafka/dlt pipeline changes,
  no LLM-assisted synthesis in v1.

## Context

- `pokemon-kafka/data/` holds ~37k telemetry events across 45 JSONL files; empirical-evidence has
  its own rollout telemetry plus `out/rollouts` / `out/harvest` genome+fitness artifacts.
- Event inventory (union of both repos): 25,430 `battle`, 7,416 `overworld`, 513 `battle_end`,
  454 `battle_outcome`, 430 `move_result`, 329 `fitness`, 193 `map_change`, 108 `moveset`,
  86 `discovery`, 11 `milestone`.
- `battle_outcome` and `move_result` were designed as labeled training rows and are usable nearly
  as-is. `docs/experiment-findings.md` establishes that genome-only training saturates; the blend
  keeps the genome task but no longer depends on it alone.

## Architecture

New module: `scripts/convert_telemetry.py` in empirical-evidence.

```
uv run python scripts/convert_telemetry.py \
  --pk-data ../pokemon-kafka/data \
  --ee-out out/ \
  --out data/sft_v3 \
  --seed 42
```

- Reads: all `*.jsonl` under `--pk-data` (recursive) and empirical-evidence telemetry; genome +
  fitness artifacts under `--ee-out` (`rollouts/`, `harvest/`).
- Writes: `data/sft_v3/corpus.jsonl`, `train.jsonl`, `valid.jsonl`, and `stats.json`
  (per-domain counts, skipped-line counts, corpus SHA-256).
- Output format: `{"messages": [{role, content}...], "domain": "<tag>"}` — the chat format the
  existing SFT trainer already consumes. No downstream format changes.

## Example generators (one function per domain)

| Domain tag | Source | Shape |
|---|---|---|
| `battle-outcome` | 454 `battle_outcome` rows | Pre-battle state (species, levels, HP, move types, enemy type, level gap) → JSON `{"win": bool, "recommendation": "fight"\|"flee"}` from the recorded label. |
| `move-choice` | 430 `move_result` rows | (a) per-row damage-bucket prediction — bucket = damage as a fraction of enemy max HP: `none` (0), `weak` (<15%), `solid` (15–40%), `heavy` (>40% or one-shot); (b) where ≥2 distinct moves were observed for the same (user species, enemy type), "which move?" → the higher-damage move. Ground-truth type effectiveness. |
| `battle-action` | ~25k `battle` turn rows | Group turns into battles per source file, ordered by `turn`, with each `battle_end` closing the current battle; keep only battles joined to a `won=true` `battle_outcome` within the same turn range (rejection sampling); emit state → the action JSON actually taken. Capped by seeded sampling. |
| `genome` | 329 `fitness` rows + `out/rollouts`, `out/harvest` | Existing 11-example prompt format, widened to all rollouts with above-median fitness per scenario. |
| `narrator` | `milestone`, `map_change`, `discovery`, `battle_end` | Sliding window of notable events → 1–2 sentence play-by-play, drawn from ≥5 seeded phrasing templates per event type. |

**Balance:** no domain exceeds 40% of the corpus (seeded down-sampling of overweight domains).
Expected total: ~2,000–5,000 examples.

## Determinism, dedup, split

- Single seeded `random.Random(seed)` for every sampling/template/split decision.
- Dedup by SHA-256 of normalized (prompt, answer) pairs.
- 90/10 train/valid split, stratified by domain.
- Invariant: same inputs + same seed → byte-identical corpus (snapshot-tested).

## Error handling

- Malformed JSONL lines: skip, count, report in `stats.json`. Never silent.
- Missing data directories: warn and continue.
- Hard failure (non-zero exit) if total examples < 500 or any domain is empty — a broken input
  assumption must not silently reproduce the 11-example situation.

## Downstream (existing pipeline, unchanged except inputs)

1. SFT: retrain LoRA on the 5090 via the existing TRL/PEFT path against `data/sft_v3`.
2. Eval gate before any publish:
   - held-out valid loss, and
   - scripted comparison vs base SmolLM3-3B on held-out rows: type-effectiveness move picks
     (`move-choice`) and win prediction (`battle-outcome`). **If the tuned model does not beat
     base, do not publish.**
3. Package: fuse adapter, update manifest (corpus SHA, per-domain counts, seed, base model).
4. Publish to HF Hub: fused checkpoint + adapter + model card. Model card is explicit that
   prompts are template-generated from game telemetry and the model is a demo-scale agent
   assistant, not a general Pokémon expert.

## Testing

- Pytest with tiny fixture JSONL files per event type under `tests/fixtures/`.
- One unit test per generator (known input → expected example).
- Dedup + stratified-split determinism tests.
- Corpus snapshot test: fixed fixtures + fixed seed → fixed corpus hash.
- Run with `uv run python -m pytest` (plain `uv run pytest` does not spawn in this repo).

## Out of scope for v1

- DPO / preference pairs.
- Kafka/dlt/Flink pipeline changes (Approach B).
- LLM-assisted paraphrasing (Approach C).
- Continuous corpus regeneration; the converter is run manually per release.
