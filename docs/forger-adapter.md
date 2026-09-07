# The Forger's data and adapters (2026-09-07)

The Forger is the expedition seat that handles what the game says back: given where a body stands,
its sprite picture and the sentence read from the screen, name what it is and what the talk yields;
read a refusal sentence as a gate class; take an operator handoff. Everything here is measured from
runs on the cartridge; no recalled game facts (pokemon-kafka `AGENTS.md`).

## Dataset — [`bdougie/pokemon-red-sft`](https://huggingface.co/datasets/bdougie/pokemon-red-sft)

`data/sft_v5u`, 27,836 rows, 25,053 train / 2,783 valid (seed 42, 10 %). Built as
`autotune.merge_corpus` of `data/sft_v4` (the 2026-09-05 corpus) and `data/sft_v5`
(`autotune.convert_telemetry --pk-root ../pokemon-kafka --pk-data ../pokemon-kafka/data/telemetry/game --max-rss-gb 40`,
11,003 rows from the expedition sink and the recorder runs after the forward-play sweep). The sink
alone does not hold v4's battle sources, so the union is what keeps the Wheelman whole.

| seat | domain | v4 | v5u |
|---|---|---:|---:|
| Forger | npc-dialogue | 848 | 1318 |
| Forger | gate-text | 65 | 82 |
| Operator | handoff | 77 | 81 |
| Wheelman | battle-outcome | 3426 | 3571 |
| Wheelman | move-choice | 9734 | 10938 |
| Wheelman | battle-action | 105 | 725 |
| Extractor | puzzle-consult | 305 | 587 |
| Narrator | narrator | 9734 | 10493 |

The splits carry `domain` and `meta` on every row — the held-out gate scores by domain, and a
first cut of the splits with messages only scored nothing (and "passed"). The card is the
dataset's `README.md`; `stats.json` carries the sha and provenance.

Upload: `hf upload bdougie/pokemon-red-sft data/sft_v5u --repo-type dataset`.

## Adapters

Both are bf16 LoRA r32 (q,k,v,o,gate,up,down) on `HuggingFaceTB/SmolLM3-3B` via
`autotune.train_sft`, gated by `autotune.eval_heldout` (tuned vs base, and vs the majority label).

### [`bdougie/smollm3-pokemon-forger-lora`](https://huggingface.co/bdougie/smollm3-pokemon-forger-lora) — Forger only

`data/sft_v5_forger` = the npc-dialogue, gate-text and handoff rows of v5u (1,481; 1,333 train,
148 valid), 3 epochs. Gate on 148 held-out rows:

| metric | base | tuned | majority |
|---|---:|---:|---:|
| npc-dialogue/body | 0.21 | **0.82** | |
| npc-dialogue/outcome | 0.33 | **0.66** | 0.49 (always "talk") |
| gate-text/gate | — | **1.00** (4/4) | |

The first Forger adapter (2026-09-05, kept at `out/sft_forger1_2026-09-05`) had body 0.73 and an
outcome head one row over the majority; the sweep's 470 new dialogue rows are what moved it.

### [`bdougie/smollm3-pokemon-red-lora`](https://huggingface.co/bdougie/smollm3-pokemon-red-lora) — all seats

Trained on v5u, 900 steps (about 0.6 epoch, 32 min on the 5090). Gate on 2,783 held-out rows:

| metric | rows | base | tuned | majority |
|---|---:|---:|---:|---:|
| battle-outcome | 349 | 0.52 | **0.99** | |
| move-choice | 1061 | 0.00 | **0.95** | |
| npc-dialogue/body | 132 | 0.21 | **0.77** | |
| npc-dialogue/outcome | 132 | 0.42 | **0.63** | 0.61 (always "talk") |
| gate-text/gate | 8 | 0.00 | **1.00** | |

The mixed adapter carries the battle seats; the Forger-only adapter is the better Forger (body 0.82 vs 0.77,
outcome 0.66 vs 0.63 - and that outcome head is the weak spot in both: two points over the majority in the
mixed one). A served, quantized copy of the Forger answered a held-out prompt with the right body and the
right JSON shape but a wrong outcome; the gate numbers are the bf16 adapter's.

## Serving

**This box (CUDA):** `scripts/package_lora.sh out/forger/sft bdougie/smollm3-pokemon-forger-lora pokemon-forger`
merges the adapter into the base, converts to GGUF, quantizes Q4_K_M, uploads
`gguf/pokemon-forger.Q4_K_M.gguf` to the adapter's repo and registers `pokemon-forger:Q4_K_M` in
the local Ollama. Then `ollama run pokemon-forger:Q4_K_M`, or point pokemon-kafka's tapes proxy at
it as a seat model.

**Other machines:** the repos are public (2026-09-07), so
`ollama pull hf.co/bdougie/smollm3-pokemon-forger-lora:Q4_K_M` does it in one line; or `hf download` the GGUF and
`ollama create pokemon-forger:Q4_K_M -f Modelfile` with `FROM ./pokemon-forger.Q4_K_M.gguf`.

Apple Silicon: fuse with mlx-lm as in `docs/serving-forest-lora.md` (the `--adapter-path` flag of
mlx-lm's server is silently broken; fuse).
