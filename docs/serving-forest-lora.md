# Serving forest-lora locally (mlx-lm + pi)

How to run the published [bdougie/smollm3-forest-lora](https://huggingface.co/bdougie/smollm3-forest-lora)
adapter as a local OpenAI-compatible API on Apple Silicon, and use it from
[pi](https://github.com/badlogic/pi-mono) or anything else that speaks `/v1/chat/completions`.

The adapter is the genome proposer this repo's `autotune` loop trained for the Viridian Forest
crossing: given a rollout situation, it responds with only a JSON genome. It's an MLX-format LoRA
(~54 MB) on base `EricFillion/smollm3-3b-mlx`, so no llama.cpp / GGUF conversion is involved —
`mlx-lm` (already in this repo's `mac` extra) is the whole serving stack.

## Quick start

```bash
scripts/serve_forest_lora.sh          # downloads, fuses (first run), serves on :8080
```

The script is idempotent: it downloads the adapter to `out/hf/smollm3-forest-lora` and fuses it
into `out/hf/smollm3-forest-fused` (~2.1 GB, quantization preserved) only if missing, then serves
the fused model. Everything lives under gitignored `out/`.

Verify:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'"$PWD"'/out/hf/smollm3-forest-fused",
  "messages": [
    {"role": "system", "content": "You tune a Pokemon Red agent'\''s battle/survival genome to cross Viridian Forest to Pewter City. Respond with ONLY the JSON genome."},
    {"role": "user", "content": "This rollout reached forest beat 1 (reward 2.0 sub-beats, 1 catchers beaten, crossed=False). Current genome: {\"hp_run_threshold\": 0.2, \"hp_heal_threshold\": 0.3}"}
  ],
  "temperature": 0, "max_tokens": 300
}' | jq -r '.choices[0].message.content'
```

Adapter behavior = immediate raw JSON genome. Base-model behavior (what you get if the adapter
was dropped, see below) = markdown-fenced JSON with an invented nested schema.

## Why we fuse instead of `--adapter-path`

**mlx-lm 0.31.3's `mlx_lm.server --adapter-path` flag is silently broken.** In
`mlx_lm/server.py`, `ModelProvider.load()` resolves the request's model name *first* and then
looks up the adapter map with the resolved path:

```python
model_path = self._model_map.get(model_path, model_path)        # "default_model" -> cli model
adapter_path = self._adapter_map.get(model_path, adapter_path)  # keyed only by "default_model" -> miss
```

`_adapter_map` only has the key `"default_model"`, so the lookup always misses and every request
(including the startup default load) serves the plain base model. Verified 2026-07-09 with two
temperature-0 requests: one relying on the CLI flag returned base-model output; one passing the
adapter per-request (`"adapters": "out/hf/smollm3-forest-lora"` in the body, which takes a
different code path) returned adapter output.

pi can't send a per-request `adapters` field, so we fuse the adapter into standalone weights
(`mlx_lm.fuse`) and serve those. No adapter resolution, nothing to drop. If a later mlx-lm release
fixes the lookup order, `--adapter-path` becomes viable again — re-run the A/B above before
trusting it.

## Using it from pi

pi is only a client — it needs this server running. The provider is registered in
`~/.pi/agent/models.json` (alongside the existing `ollama` provider):

```json
"mlx": {
  "baseUrl": "http://127.0.0.1:8080/v1",
  "api": "openai-completions",
  "apiKey": "dummy",
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
  "models": [
    {
      "id": "/Users/bdougie/code/pcc-labs/empirical-evidence/out/hf/smollm3-forest-fused",
      "name": "smollm3-forest-lora (local MLX)",
      "reasoning": false,
      "input": ["text"],
      "contextWindow": 65536,
      "maxTokens": 4096,
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
    }
  ]
}
```

Gotchas baked into that config:

- **`id` is the fused model's absolute path, and it must stay that way.** mlx_lm.server loads
  whatever the request's `model` field names. The absolute path matches what the serve script
  loaded (no reload), and even cold-loads correctly if the server was started bare. A friendlier
  alias would make the server try to resolve it as a HF repo and fail. (`"default_model"` also
  works but serves whatever the server happened to start with.)
- `apiKey: "dummy"` — pi hides models whose provider has no auth; local servers ignore the value.
- `compat` flags — mlx_lm.server rejects the `developer` role and `reasoning_effort` that pi
  sends to reasoning-capable models.
- The file hot-reloads when you open `/model` in pi; no restart needed.

Run it:

```bash
pi --provider mlx --model "smollm3-forest-lora" -p --no-tools --no-session \
  --system-prompt "You tune a Pokemon Red agent's battle/survival genome to cross Viridian Forest to Pewter City. Respond with ONLY the JSON genome." \
  'This rollout reached forest beat 1 (reward 2.0 sub-beats, 1 catchers beaten, crossed=False). Current genome: {"hp_run_threshold": 0.2, "hp_heal_threshold": 0.3}'
```

This model exists for the speedrun: it's the Nudge-step genome proposer, and the endpoint above
is how run tooling queries it for battle/survival params between rollouts. pi is just a
convenient interactive client for inspecting the proposer's behavior. Always pass the genome
system prompt — pi's default coding-agent system prompt fights the adapter's training (a
proof-of-concept trained on 4 examples / 8 iterations that emits genome JSON).

## Troubleshooting

- **Server log:** `out/hf/mlx_server.log` if started via the setup notes; the script logs to
  stdout — run it under `nohup`/tmux to keep it alive.
- **Port in use:** `scripts/serve_forest_lora.sh 8081` (update `baseUrl` in models.json to match).
- **Model missing in pi's `/model`:** provider needs *some* `apiKey` value; check
  `pi --list-models | grep mlx`.
- **Output looks like base SmolLM3 (markdown fences, `<think>` tags):** you're not hitting the
  fused weights — check the `model` id matches the fused path exactly, and that you didn't fall
  back to `--adapter-path` serving (broken, see above).
- **CUDA box:** mlx-lm won't install on Linux. Serve the non-MLX base + adapter with vLLM
  (`vllm serve HuggingFaceTB/SmolLM3-3B --enable-lora --lora-modules forest=<adapter>`), noting
  the published adapter is MLX-format — retrain/export from the CUDA backend's PEFT output
  instead of using the HF repo directly.
