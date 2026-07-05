---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - pokemon
  - game-agent
  - sft
pretty_name: Pokemon Red Telemetry SFT (multi-domain)
---

# Pokemon Red Telemetry SFT corpus (sft_v3)

590 chat-format SFT examples generated deterministically (seed 42) from ~72k `pokemon.game.v1`
telemetry events emitted by an autonomous Pokemon Red agent
([pokemon-kafka](https://github.com/pcc-labs/pokemon-kafka)) and its training loop
(empirical-evidence). Labels for the battle domains come from game RAM, not annotation.

## Domains

| domain | examples | source | label |
|---|---|---|---|
| narrator | 236 | milestone / map_change / discovery / battle_end events | template-generated one-sentence play-by-play (synthetic style labels) |
| battle-action | 144 | per-turn `battle` events of battles that were won | the action JSON the agent actually took (rejection-sampled on outcome) |
| move-choice | 97 | `move_result` rows | damage bucket vs enemy max HP; best-move picks where ≥2 moves observed per matchup |
| battle-outcome | 72 | `battle_outcome` rows | recorded win/loss → fight/flee recommendation |
| genome | 41 | rollout genome+fitness artifacts | above-median-fitness survival-parameter JSON per scenario |

## Files

- `corpus.jsonl` — all examples, each `{"messages": [...], "domain": ...}`
- `train.jsonl` (533) — `domain` stripped, feeds TRL `SFTTrainer` directly
- `valid.jsonl` (57) — stratified 10% split, keeps `domain` for per-domain eval
- `stats.json` — per-domain counts, skipped-line count, seed, corpus SHA-256

## Generation

```
uv run python -m autotune.convert_telemetry \
  --pk-data ../pokemon-kafka/data --ee-data data/telemetry \
  --out data/sft_v3 --seed 42
```

Deterministic: same inputs + same seed → byte-identical corpus
(`corpus_sha256` in `stats.json`; converter source: `autotune/convert_telemetry.py`).

## Known limitations

- Early-game Pokemon Red only (Pallet Town → Pewter City area); narrow species/type coverage.
- ~55% of battle-action prompts contain raw `#NN` species codes where the RAM reader had no
  resolved name (e.g. `#71`); the model sees these as opaque identifiers.
- Narrator labels are template-generated (5 seeded phrasings per event type) — stylistic, not
  human commentary.
- Heavy dedup: the agent grinds repetitive early-game states, so 72k raw events collapse to 590
  unique examples.
