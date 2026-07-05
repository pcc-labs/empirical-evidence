---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM3-3B
tags:
  - pokemon
  - game-agent
  - lora
  - smollm3
---

# SmolLM3-3B Pokemon Red Agent (multi-task)

SmolLM3-3B LoRA-fine-tuned (fused) on telemetry from an autonomous Pokemon Red agent
([pokemon-kafka](https://github.com/pcc-labs/pokemon-kafka)) and its Try→Check→Reward→Nudge
training loop (empirical-evidence). Battle-domain labels come from game RAM telemetry, not
human annotation.

Three tasks in one model, all plain chat:

- **Battle advisor** — pre-battle state → `{"win": bool, "recommendation": "fight"|"flee"}`;
  move damage buckets and best-move picks (ground-truth type effectiveness).
- **Genome proposer** — scenario + fitness summary → survival-parameter JSON for the
  pokemon-kafka agent.
- **Narrator** — game event JSON → one-sentence play-by-play for a stream overlay.

## Held-out eval (vs base SmolLM3-3B, greedy, 64 new tokens)

| Domain | Base | Tuned |
|---|---|---|
| battle-outcome (win prediction) | 0.00 | 1.00 |
| move-choice (type effectiveness) | 0.00 | 0.56 |

Base scores 0 because untuned SmolLM3-3B does not emit the requested JSON shape. The held-out
set is small (57 rows, ~17 gated), so read these as a smoke-level gate, not a benchmark.

## Training data

590 examples (533 train / 57 valid) generated deterministically (seed 42) from ~72k
`pokemon.game.v1` telemetry events by `autotune/convert_telemetry.py`. Per-domain counts,
corpus SHA-256, and the exact converter command are in the companion dataset repo's
`stats.json`/README. The fused checkpoint's `manifest.json` records the base model, adapter
path, and train-data fingerprint (`d36161f1c140bedf`).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID", device_map="auto")
msgs = [
    {"role": "system", "content": "You are the battle advisor for a Pokemon Red agent. "
     "Answer with only the requested JSON."},
    {"role": "user", "content": 'Battle start.\nYour Pokemon: Charmander (lv 6, HP 21/21), '
     'move types: normal, fire.\nEnemy: Weedle (lv 3, bug type). Level gap: +3. '
     'Healing available: no.\nWill the agent win this battle, and should it fight or flee? '
     'Respond with JSON {"win": bool, "recommendation": "fight"|"flee"}.'},
]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids.to(model.device), max_new_tokens=48, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# {"win": true, "recommendation": "fight"}
```

The LoRA adapter (rank per `adapter/adapter_config.json`) is also included under `adapter/`
for use with PEFT on the base model.

## Limitations

- Trained only on early-game Pokemon Red states (Pallet Town → Pewter City); species/type
  coverage is narrow.
- ~55% of battle-action training prompts contain raw `#NN` species codes where the RAM reader
  had no resolved name; the model treats them as opaque identifiers.
- Narrator labels are synthetic (seeded template pools) — narration is stylistic, not learned
  commentary.
- The genome task is agent-specific; genome JSON is meaningless outside the pokemon-kafka agent.
- 533 training examples is demo scale; final train loss 0.069 indicates substantial
  memorization of the telemetry distribution. This is a demo-scale agent assistant, not a
  general Pokemon expert.
