# Domain-Tagged LoRA Training Corpus + Per-Domain Forest Checkpoint Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one LoRA adapter on a domain-tagged union corpus (battles / navigation / discovery) and benchmark every checkpoint with the dense forest reward broken into per-domain sub-scores. (GitHub issue #10 is the spec of record.)

**Architecture:** `forest_story` gains a beat→domain mapping plus `domain_scores` / `pair_domains` pure functions. Both harvests (`forest_harvest` = survival/battle pairs, `harvest` = map-grained nav pairs) tag their improvement pairs and persist a full tagged `corpus.jsonl` alongside the stripped `train.jsonl`/`valid.jsonl`. A new `merge_corpus` CLI unions corpora with a loud failure on empty inputs and a per-domain census. A new `forest_benchmark` sweeps every LoRA checkpoint through the genome-driven forest follower and reports a per-domain trend table + plot, pairing 1:1 with `weights_viz`. Checkpoint discovery is extended to the PEFT/cuda layout so the sweep runs on both backends.

**Tech Stack:** Python 3.12, `uv`, pytest, ruff (E,F,I,W, line 100), matplotlib (Agg), safetensors, HF/TRL/PEFT (cuda) or mlx-lm (mac), PyBoy via pokemon-kafka.

## Global Constraints

- Branch: `feat/lora-weight-viz` (open PR #8). Do not push unless asked.
- Domain taxonomy (exact, from issue #10): beat 1 → `nav`, beat 2 → `discovery`, beat 3 → `battle`, beat 4 → `battle`, beat 5 → `discovery`, beat 6 → `discovery`, beat 7 → `discovery`, beat 8 → `nav`. Domain names are exactly `"nav"`, `"battle"`, `"discovery"`.
- Corpus row shape (tagged): `{"messages": [...], "domains": ["battle", ...]}`. Train/valid row shape (stripped): `{"messages": [...]}` only — domain tags must never reach the training chat data.
- `merge_corpus` must fail LOUDLY (non-zero exit, clear message) when any input corpus is missing or empty.
- Item beats (2, 5, 7) score 0 until pokemon-kafka emits `bag_count` — accepted; do NOT change pokemon-kafka (out of scope per issue and AGENTS.md).
- Out of scope: per-domain adapters, pokemon-kafka changes.
- Use `uv` for everything. Tests: `uv run python -m pytest -q` (NOT `uv run pytest` — that fails to spawn in this venv). Lint: `uv run ruff check autotune/ tests/` (untracked `scratch_*.py` at repo root are known-dirty; never lint or fix them).
- Pure logic is unit-tested; GPU/emulator/subprocess drivers are `# pragma: no cover` and exercised by the smoke run (AGENTS.md convention).
- Line length 100; import order per ruff `I`.
- Commit after each task with a conventional-commit message.

## Reference: key existing interfaces (already on the branch after the merge)

- `autotune/forest_story.py`: `FOREST_BEATS: tuple[ForestBeat, ...]` (`ForestBeat(beat_id, name)`, ids 1..8), `ForestVerdict(furthest_beat, furthest_beat_name, beats_passed, per_beat, reward, crossed, signals)`, `score_forest(events) -> ForestVerdict`, `extract_forest_signals(events) -> ForestSignals`.
- `autotune/nudge_sft.py`: `Winner(params, verdict)`, `build_pair_example(source: Winner, target_params: dict, story: Story) -> dict`, `build_dataset(winners, story)`, `assemble_corpus(winners_by_state, story)`, `split_train_valid(examples, valid_frac=0.2, seed=42)`, `write_jsonl(path, rows)`, `write_sft_data(sft_dir, examples, valid_frac=0.2, seed=42) -> (train_path, valid_path)`; forest half: `_FOREST_SYSTEM`, `ForestWinner(params, verdict, fitness)`, `_forest_rank(w)`, `build_forest_mutation_prompt(params, verdict) -> str`, `build_forest_pair_example(source: ForestWinner, target_params: dict) -> dict`, `build_forest_dataset(winners)`, `assemble_forest_corpus(winners_by_state)`.
- `autotune/forest_harvest.py`: `harvest(cfg, in_state, state_label, genomes, max_steps, worldmap_in, out_dir) -> dict` — calls `follow_once` per genome, `assemble_forest_corpus({state_label: winners})`, `write_sft_data(out_dir, examples)`.
- `autotune/harvest.py`: `run_harvest(cfg, state_paths, n_genomes, max_turns, seed)` — map-grained nav harvest; writes via `write_sft_data(cfg.storage.sft_dir, examples, seed=seed)`. `resolve_states(spec) -> list[str]`.
- `autotune/forest_follow.py`: `follow_once(rom, in_state, genome, max_steps=1500, worldmap_in=None, worldmap_out=None, out_state=None, shot=None, route=None) -> dict` with keys `events`, `fitness`, `reward`, `crossed`, ...; `ROUTE_DEFAULT = str(Path(__file__).parent / "routes" / "forest_cross_path.json")`. NOTE: without `route`, it uses legacy BEATS nav that wedges before the exit; `main()` defaults to `ROUTE_DEFAULT` when the file exists but `follow_once` itself does not.
- `autotune/weights_viz.py`: `discover_checkpoints(adapter_dir) -> list[tuple[str, Path]]` — currently mlx layout only (`NNNN_adapters.safetensors` + final `adapters.safetensors`); `load_lora_deltas(path)` already normalizes both mlx and PEFT tensor keys.
- `autotune/benchmark.py`: `stage_checkpoint(adapter_dir, label, ckpt_path) -> Path` (stages an mlx numbered checkpoint as a loadable adapter dir; `"final"` returns `adapter_dir` as-is), `format_trend`, `run_benchmark` (map-grained; uses `discover_checkpoints` + `stage_checkpoint`).
- `autotune/generate.py`: `make_proposer(cfg, adapter_dir=None) -> Callable[[str], str]`; `_generate_cuda(cfg, prompt, adapter_path)` and `_generate_mlx(cfg, prompt, adapter_path)` both hardcode the map-story system prompt `nudge_sft._SYSTEM`.
- `autotune/nudge_steer.py`: `parse_genome_response(text) -> dict | None` (clamped genome or None).
- `autotune/train_sft.py`: `train(cfg, data_dir, iters=None) -> Path`; cuda via TRL `SFTTrainer` (checkpoints land as `checkpoint-N/` dirs), mlx via `mlx_lm lora` config from `build_lora_config(cfg, data_dir, adapter_dir, iters)` (checkpoints land as `NNNN_adapters.safetensors`); `load_jsonl(path)`.
- `autotune/config.py`: `cfg.storage.sft_dir == ./data/sft`, `cfg.storage.adapter_dir == ./out/sft`, `cfg.env.rom_path`, `cfg.backend` (`"cuda"` on this machine).
- Test helpers in `tests/test_forest_story.py`: `_ow(map_id, bag=None, turn=0)`, `_trainer_win(turn=0)`, `_sign(text=..., turn=0)` build synthetic telemetry events.

## File Structure

- Modify: `autotune/forest_story.py` — add `DOMAINS`, `BEAT_DOMAINS`, `domain_scores()`, `pair_domains()` (Task 1)
- Modify: `autotune/nudge_sft.py` — tag forest + map pairs, `write_corpus()`, strip tags in `write_sft_data()` (Tasks 2–3)
- Modify: `autotune/forest_harvest.py` — persist tagged `corpus.jsonl`, pass the canonical route (Task 2)
- Modify: `autotune/harvest.py` — persist tagged `corpus.jsonl` (Task 3)
- Create: `autotune/merge_corpus.py` — union CLI + census (Task 4)
- Modify: `autotune/weights_viz.py`, `autotune/benchmark.py`, `autotune/train_sft.py` — PEFT checkpoint layout + save cadence (Task 5)
- Create: `autotune/forest_benchmark.py`; modify `autotune/generate.py` (system-prompt threading) (Task 6)
- Tests: `tests/test_forest_story.py`, `tests/test_nudge_sft.py`, `tests/test_harvest.py`, `tests/test_merge_corpus.py` (new), `tests/test_weights_viz.py`, `tests/test_benchmark.py`, `tests/test_train_sft.py` (new), `tests/test_forest_benchmark.py` (new)

---

### Task 1: Beat→domain mapping + `domain_scores` / `pair_domains` in `forest_story.py`

**Files:**
- Modify: `autotune/forest_story.py`
- Test: `tests/test_forest_story.py`

**Interfaces:**
- Consumes: existing `FOREST_BEATS`, `ForestVerdict`, `score_forest`.
- Produces (later tasks rely on these exact names):
  - `DOMAINS: tuple[str, ...] = ("nav", "battle", "discovery")`
  - `BEAT_DOMAINS: dict[int, str]` — the exact taxonomy from Global Constraints
  - `domain_scores(verdict: ForestVerdict) -> dict[str, int]`
  - `pair_domains(source: ForestVerdict, target: ForestVerdict) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_forest_story.py` (reuse the existing `_ow`/`_trainer_win`/`_sign` helpers; extend the module import line):

```python
from autotune.forest_story import (
    BEAT_DOMAINS,
    DOMAINS,
    FOREST_BEATS,
    domain_scores,
    extract_forest_signals,
    pair_domains,
    score_forest,
)


def test_every_beat_has_exactly_one_domain():
    assert set(BEAT_DOMAINS) == {b.beat_id for b in FOREST_BEATS}
    assert set(BEAT_DOMAINS.values()) <= set(DOMAINS)


def test_domain_scores_split_by_domain():
    # enter + 2 catcher wins + sign + exit; no bag_count in telemetry
    v = score_forest([_ow(51), _trainer_win(), _trainer_win(), _sign(), _ow(13)])
    assert domain_scores(v) == {"nav": 2, "battle": 2, "discovery": 1}


def test_domain_scores_zero_outside_forest():
    v = score_forest([_ow(12)])
    assert domain_scores(v) == {"nav": 0, "battle": 0, "discovery": 0}


def test_pair_domains_names_improved_domains():
    weak = score_forest([_ow(51)])
    strong = score_forest([_ow(51), _trainer_win(), _ow(13)])
    assert pair_domains(weak, strong) == ("nav", "battle")


def test_pair_domains_tie_falls_back_to_battle():
    # Same beats on both sides: the improvement was a survival tiebreak
    # (trainer_wins/turns in _forest_rank), which is battle domain.
    a = score_forest([_ow(51)])
    b = score_forest([_ow(51)])
    assert pair_domains(a, b) == ("battle",)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_forest_story.py -q`
Expected: ImportError (`BEAT_DOMAINS` etc. not defined).

- [ ] **Step 3: Implement** — add to `autotune/forest_story.py` after the `FOREST_BEATS` tuple / gate-removal NOTE comment:

```python
# --------------------------------------------------------------------------- #
# Behavior domains (issue #10): each beat exercises exactly one domain.        #
# --------------------------------------------------------------------------- #

DOMAINS: tuple[str, ...] = ("nav", "battle", "discovery")

# nav = map transitions (enter/exit), battle = the bug-catcher fights, discovery = items + sign.
BEAT_DOMAINS: dict[int, str] = {
    1: "nav",
    2: "discovery",
    3: "battle",
    4: "battle",
    5: "discovery",
    6: "discovery",
    7: "discovery",
    8: "nav",
}


def domain_scores(verdict: ForestVerdict) -> dict[str, int]:
    """Per-domain count of beats reached: nav 0..2, battle 0..2, discovery 0..4. Pure."""
    scores = dict.fromkeys(DOMAINS, 0)
    for beat, passed in zip(FOREST_BEATS, verdict.per_beat):
        if passed:
            scores[BEAT_DOMAINS[beat.beat_id]] += 1
    return scores


def pair_domains(source: ForestVerdict, target: ForestVerdict) -> tuple[str, ...]:
    """Domains an improvement pair teaches: those where the target out-scored the source.

    When no domain differs, the pair was ranked up by ``_forest_rank``'s survival tiebreaks
    (trainer_wins, fewer turns) — battle-domain levers — so tag it ``("battle",)`` rather than
    dropping it untagged.
    """
    src, tgt = domain_scores(source), domain_scores(target)
    improved = tuple(d for d in DOMAINS if tgt[d] > src[d])
    return improved or ("battle",)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_forest_story.py -q`
Expected: all pass (existing + 5 new).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check autotune/forest_story.py tests/test_forest_story.py
git add autotune/forest_story.py tests/test_forest_story.py
git commit -m "feat(forest_story): beat->domain taxonomy with domain_scores and pair_domains"
```

---

### Task 2: Domain-tag forest SFT pairs; persist tagged `corpus.jsonl`; strip tags from train/valid

**Files:**
- Modify: `autotune/nudge_sft.py`
- Modify: `autotune/forest_harvest.py`
- Test: `tests/test_nudge_sft.py`

**Interfaces:**
- Consumes: `pair_domains` from Task 1; existing `ForestWinner`, `_forest_rank`, `write_jsonl`, `split_train_valid`.
- Produces:
  - `build_forest_pair_example(source: ForestWinner, target: ForestWinner) -> dict` — **signature change**: takes the target `ForestWinner` (was `target_params: dict`) so it can compute `pair_domains(source.verdict, target.verdict)`. Returned dict gains `"domains": list[str]`. Its only caller is `build_forest_dataset` (plus tests).
  - `write_corpus(path: Path, examples: list[dict]) -> Path` — writes the TAGGED corpus JSONL.
  - `write_sft_data(...)` unchanged signature, but train/valid rows are stripped to `{"messages": ...}` only.
  - `forest_harvest.harvest(...)` summary dict gains `"corpus": str(path)`; harvest passes the canonical route to `follow_once`.

- [ ] **Step 1: Write the failing tests** — in `tests/test_nudge_sft.py`, first read the file to see existing forest fixtures/imports, then add (adapting helper names only if identical ones already exist):

```python
from autotune.forest_story import score_forest
from autotune.nudge_sft import (
    ForestWinner,
    build_forest_dataset,
    build_forest_pair_example,
    write_corpus,
    write_sft_data,
)


def _fw(params, events, turns=500):
    return ForestWinner(params=params, verdict=score_forest(events), fitness={"turns": turns})


def _enter():
    return {"event_type": "overworld", "turn": 0, "data": {"map_id": 51}}


def _twin():
    return {"event_type": "battle_outcome", "turn": 1, "data": {"battle_type": 2, "won": True}}


def _exit():
    return {"event_type": "overworld", "turn": 2, "data": {"map_id": 13}}


def test_forest_pair_example_carries_pair_domains():
    weak = _fw({"hp_run_threshold": 0.6}, [_enter()])
    strong = _fw({"hp_run_threshold": 0.1}, [_enter(), _twin(), _exit()])
    ex = build_forest_pair_example(weak, strong)
    assert ex["domains"] == ["nav", "battle"]
    assert set(ex) == {"messages", "domains"}


def test_forest_dataset_examples_are_tagged():
    weak = _fw({"hp_run_threshold": 0.6}, [_enter()])
    strong = _fw({"hp_run_threshold": 0.1}, [_enter(), _twin(), _exit()])
    examples = build_forest_dataset([weak, strong])
    assert examples and all(isinstance(e["domains"], list) and e["domains"] for e in examples)


def test_write_corpus_keeps_domains(tmp_path):
    import json

    rows = [{"messages": [{"role": "user", "content": "x"}], "domains": ["battle"]}]
    path = write_corpus(tmp_path / "corpus.jsonl", rows)
    on_disk = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert on_disk == rows


def test_write_sft_data_strips_domains(tmp_path):
    import json

    rows = [
        {"messages": [{"role": "user", "content": str(i)}], "domains": ["nav"]}
        for i in range(5)
    ]
    train_path, valid_path = write_sft_data(tmp_path, rows)
    for p in (train_path, valid_path):
        for ln in p.read_text().splitlines():
            assert set(json.loads(ln)) == {"messages"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_nudge_sft.py -q`
Expected: ImportError on `write_corpus`, then assertion/type failures.

- [ ] **Step 3: Implement in `autotune/nudge_sft.py`**

Import the taxonomy (extend the existing `forest_story` import line):

```python
from autotune.forest_story import ForestVerdict, pair_domains
```

Replace `build_forest_pair_example` (and update its docstring):

```python
def build_forest_pair_example(source: ForestWinner, target: ForestWinner) -> dict:
    """One chat example: improve ``source``'s genome -> the stronger ``target``'s genome.

    Tagged with the behavior domains the improvement spans (``forest_story.pair_domains``) so a
    union corpus can be censused or filtered per domain; ``write_sft_data`` strips the tag before
    training.
    """
    answer = clamp_params(target.params)
    return {
        "messages": [
            {"role": "system", "content": _FOREST_SYSTEM},
            {
                "role": "user",
                "content": build_forest_mutation_prompt(source.params, source.verdict),
            },
            {"role": "assistant", "content": json.dumps(answer)},
        ],
        "domains": list(pair_domains(source.verdict, target.verdict)),
    }
```

In `build_forest_dataset`, change the call `build_forest_pair_example(w, target.params)` → `build_forest_pair_example(w, target)`.

Add after `write_jsonl`:

```python
def write_corpus(path: Path, examples: list[dict]) -> Path:
    """Persist the TAGGED corpus (messages + domains) — the merge/census input, not train data."""
    path = Path(path)
    write_jsonl(path, examples)
    return path


def _strip_tags(rows: list[dict]) -> list[dict]:
    """Training rows carry ONLY messages — domain tags are corpus metadata, not chat turns."""
    return [{"messages": r["messages"]} for r in rows]
```

In `write_sft_data`, strip at the write site:

```python
    write_jsonl(train_path, _strip_tags(train))
    # MLX-LM still expects valid.jsonl to exist; mirror train when too few examples to split.
    write_jsonl(valid_path, _strip_tags(valid or train))
```

- [ ] **Step 4: Modify `autotune/forest_harvest.py`**

Extend the import: `from autotune.nudge_sft import ForestWinner, assemble_forest_corpus, write_corpus, write_sft_data` and add `from autotune.forest_follow import ROUTE_DEFAULT, follow_once`.

In `harvest()`, pass the canonical route so crossings can actually complete (mirrors `forest_follow.main`'s default; without it the legacy BEATS nav wedges before the exit):

```python
    route = ROUTE_DEFAULT if Path(ROUTE_DEFAULT).exists() else None
    for i, genome in enumerate(genomes):
        result = follow_once(
            rom, in_state, genome, max_steps=max_steps, worldmap_in=worldmap_in, route=route
        )
```

After `examples = assemble_forest_corpus(...)` and the empty-guard, persist the tagged corpus next to the train data:

```python
    corpus_path = write_corpus(out_dir / "corpus.jsonl", examples)
    train_path, valid_path = write_sft_data(out_dir, examples)
    print(f"[harvest] {len(examples)} SFT pairs from {len(winners)} runs "
          f"(rewards seen: {rewards}) -> {train_path}")
    return {
        "examples": len(examples),
        "rewards_seen": rewards,
        "crossed_any": any(w.verdict.crossed for w in winners),
        "corpus": str(corpus_path),
        "train": str(train_path),
        "valid": str(valid_path),
    }
```

- [ ] **Step 5: Run the full suite** (other tests may assert the old example shape — update any that break to expect the `domains` key)

Run: `uv run python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check autotune/ tests/
git add autotune/nudge_sft.py autotune/forest_harvest.py tests/test_nudge_sft.py
git commit -m "feat(nudge_sft): domain-tag forest pairs, persist tagged corpus.jsonl, strip tags from train/valid"
```

---

### Task 3: Tag map-grained (nav) harvest pairs; persist its `corpus.jsonl`

**Files:**
- Modify: `autotune/nudge_sft.py` (map half), `autotune/harvest.py`
- Test: `tests/test_nudge_sft.py`, `tests/test_harvest.py`

**Interfaces:**
- Consumes: `write_corpus`, `_strip_tags` from Task 2.
- Produces: `build_pair_example(source: Winner, target_params: dict, story: Story, domains: tuple[str, ...] = ("nav",)) -> dict` — returned dict gains `"domains": list(domains)`. Default `("nav",)` because the map-grained story reward is map progress and `harvest.build_genome_population` varies only `NAV_PARAM_KEYS`. `run_harvest` writes `cfg.storage.sft_dir / "corpus.jsonl"` and its summary gains `"corpus_path"`.

- [ ] **Step 1: Write the failing tests** — add to `tests/test_nudge_sft.py` (read the file first; it already has map-story fixtures for `build_pair_example` — reuse its existing `Winner`/`Story` fixtures rather than inventing new ones):

```python
def test_map_pair_example_tagged_nav_by_default(...existing fixtures...):
    ex = build_pair_example(source_winner, target_params, story)
    assert ex["domains"] == ["nav"]
    assert set(ex) == {"messages", "domains"}
```

(Adapt fixture names to what the file already defines — the assertion lines are the requirement.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_nudge_sft.py -q`
Expected: KeyError/assertion failure on `domains`.

- [ ] **Step 3: Implement** — in `autotune/nudge_sft.py`, change `build_pair_example`:

```python
def build_pair_example(
    source: Winner, target_params: dict, story: Story, domains: tuple[str, ...] = ("nav",)
) -> dict:
    """One MLX-LM chat example: improve ``source``'s genome -> the better ``target_params``.

    The user turn is the exact prompt the loop uses at inference (``build_mutation_prompt``); the
    assistant turn is the flat target genome JSON (parseable by ``parse_genome_response``). Tagged
    ``("nav",)`` by default: the map-grained story reward is map progress, and the harvest
    population varies only navigation params.
    """
    answer = clamp_params(target_params)
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": build_mutation_prompt(source.params, source.verdict, story),
            },
            {"role": "assistant", "content": json.dumps(answer)},
        ],
        "domains": list(domains),
    }
```

In `autotune/harvest.py`, extend the import to include `write_corpus`, and in `run_harvest` after `examples = assemble_corpus(...)`:

```python
    corpus_path = write_corpus(cfg.storage.sft_dir / "corpus.jsonl", examples)
    train_path, valid_path = write_sft_data(cfg.storage.sft_dir, examples, seed=seed)
    return {
        "states": len(state_paths),
        "genomes": len(population),
        "examples": len(examples),
        "corpus_path": str(corpus_path),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
```

- [ ] **Step 4: Run the full suite** (update any test asserting the old `build_pair_example` shape)

Run: `uv run python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check autotune/ tests/
git add autotune/nudge_sft.py autotune/harvest.py tests/
git commit -m "feat(harvest): tag map-grained pairs as nav, persist tagged corpus.jsonl"
```

---

### Task 4: `merge_corpus` CLI — union corpora, loud failure on empty, domain census

**Files:**
- Create: `autotune/merge_corpus.py`
- Test: `tests/test_merge_corpus.py` (new)

**Interfaces:**
- Consumes: `write_corpus`, `write_sft_data` (nudge_sft), `load_jsonl` (train_sft).
- Produces:
  - `load_corpus(path: Path) -> list[dict]` — `SystemExit` on missing/empty
  - `merge_corpora(corpora: list[list[dict]]) -> list[dict]` — concat, dedupe by messages
  - `domain_census(examples: list[dict]) -> dict[str, int]`
  - CLI: `python -m autotune.merge_corpus --inputs A/corpus.jsonl B/corpus.jsonl --out-dir data/sft_union [--valid-frac 0.2] [--seed 42]`

- [ ] **Step 1: Write the failing tests** — `tests/test_merge_corpus.py`:

```python
"""Tests for the corpus union CLI's pure seams (autotune/merge_corpus.py)."""

from __future__ import annotations

import json

import pytest

from autotune.merge_corpus import domain_census, load_corpus, merge_corpora


def _ex(content: str, domains: list[str]) -> dict:
    return {"messages": [{"role": "user", "content": content}], "domains": domains}


def test_merge_unions_and_dedupes_identical_messages():
    a = [_ex("one", ["nav"]), _ex("two", ["battle"])]
    b = [_ex("two", ["battle"]), _ex("three", ["discovery"])]
    merged = merge_corpora([a, b])
    assert [e["messages"][0]["content"] for e in merged] == ["one", "two", "three"]


def test_census_counts_each_tag_and_untagged():
    merged = [_ex("a", ["nav", "battle"]), _ex("b", ["battle"]), {"messages": []}]
    assert domain_census(merged) == {"nav": 1, "battle": 2, "untagged": 1}


def test_load_corpus_missing_is_loud(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        load_corpus(tmp_path / "nope.jsonl")


def test_load_corpus_empty_is_loud(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="empty"):
        load_corpus(p)


def test_load_corpus_roundtrip(tmp_path):
    p = tmp_path / "corpus.jsonl"
    rows = [_ex("a", ["nav"])]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_corpus(p) == rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_merge_corpus.py -q`
Expected: ModuleNotFoundError `autotune.merge_corpus`.

- [ ] **Step 3: Implement `autotune/merge_corpus.py`**

```python
"""Union domain-tagged SFT corpora into one training set.

Each harvest persists its full tagged corpus as ``corpus.jsonl`` (rows:
``{"messages": [...], "domains": [...]}``) — the forest harvest contributes battle/discovery
pairs, the map-grained harvest contributes nav pairs. This CLI unions those corpora, fails
LOUDLY when an input is missing or empty (a silent empty union would train the LoRA on nothing),
prints the per-domain census, and writes the merged tagged ``corpus.jsonl`` plus the stripped
``train.jsonl`` / ``valid.jsonl`` that ``train_sft`` consumes.

``load_corpus`` / ``merge_corpora`` / ``domain_census`` are pure and unit-tested; the CLI wrapper
is exercised by the smoke run (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotune.nudge_sft import write_corpus, write_sft_data
from autotune.train_sft import load_jsonl


def load_corpus(path: Path) -> list[dict]:
    """Read one tagged corpus JSONL; SystemExit on missing/empty (loud, per issue #10)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[merge_corpus] {path} does not exist — run its harvest first.")
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"[merge_corpus] {path} is empty — its harvest found no gradient.")
    return rows


def merge_corpora(corpora: list[list[dict]]) -> list[dict]:
    """Union corpora in order, dropping exact-duplicate examples (same messages), keeping first."""
    seen: set[str] = set()
    merged: list[dict] = []
    for rows in corpora:
        for row in rows:
            key = json.dumps(row.get("messages"), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def domain_census(examples: list[dict]) -> dict[str, int]:
    """Examples per domain tag (an example with N tags counts once per tag; no tag = untagged)."""
    census: dict[str, int] = {}
    for ex in examples:
        for domain in ex.get("domains") or ["untagged"]:
            census[domain] = census.get(domain, 0) + 1
    return census


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Union domain-tagged SFT corpora for train_sft.")
    p.add_argument("--inputs", nargs="+", required=True, help="corpus.jsonl paths to union")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    merged = merge_corpora([load_corpus(Path(x)) for x in args.inputs])
    census = domain_census(merged)
    corpus_path = write_corpus(args.out_dir / "corpus.jsonl", merged)
    train_path, _valid_path = write_sft_data(args.out_dir, merged, args.valid_frac, args.seed)
    print(f"[merge_corpus] {len(merged)} examples, census {json.dumps(census)}")
    print(f"[merge_corpus] tagged corpus -> {corpus_path}; train data -> {train_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_merge_corpus.py -q`
Expected: 5 pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check autotune/merge_corpus.py tests/test_merge_corpus.py
git add autotune/merge_corpus.py tests/test_merge_corpus.py
git commit -m "feat(merge_corpus): union tagged corpora with loud empty-input failure and domain census"
```

---

### Task 5: Cross-backend checkpoint discovery + cuda/mlx save cadence

The benchmark sweep must run on this Linux/cuda box, but `discover_checkpoints` only knows the mlx layout (`NNNN_adapters.safetensors`). TRL/PEFT training writes `checkpoint-N/` dirs with `adapter_model.safetensors` + `adapter_config.json` (already valid PEFT adapter dirs), and the final adapter lands in the adapter dir itself as `adapter_model.safetensors`.

**Files:**
- Modify: `autotune/weights_viz.py` (`discover_checkpoints` learns the PEFT layout)
- Modify: `autotune/benchmark.py` (add `discover_adapter_dirs`; `run_benchmark` uses it)
- Modify: `autotune/train_sft.py` (`--save-steps` so short smoke runs still produce numbered checkpoints)
- Test: `tests/test_weights_viz.py`, `tests/test_benchmark.py`, `tests/test_train_sft.py` (new)

**Interfaces:**
- Consumes: existing `discover_checkpoints`, `stage_checkpoint`, `build_lora_config`.
- Produces:
  - `weights_viz.discover_checkpoints(adapter_dir)` — unchanged signature; now ALSO returns PEFT entries: numbered `("N", <adapter_dir>/checkpoint-N/adapter_model.safetensors)` and final `("final", <adapter_dir>/adapter_model.safetensors)`. mlx layout takes precedence for numbered entries; `adapters.safetensors` takes precedence for final.
  - `benchmark.discover_adapter_dirs(adapter_dir: Path) -> list[tuple[str, Path]]` — labels mapped to READY-TO-LOAD adapter dirs (staging mlx numbered files, passing PEFT checkpoint dirs through, `"final"` → `adapter_dir`).
  - `train_sft.build_lora_config(cfg, data_dir, adapter_dir, iters=None, save_steps=None)` — sets `config["save_every"] = save_steps` when given; `train(cfg, data_dir, iters=None, save_steps=None)`; `_train_cuda(..., save_steps)` adds `save_strategy="steps", save_steps=save_steps` to `TRLSFTConfig` only when given; CLI gains `--save-steps`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_weights_viz.py` (read it first; reuse its existing fixture style):

```python
def test_discover_checkpoints_peft_layout(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-2" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    ckpts = discover_checkpoints(tmp_path)
    assert [label for label, _ in ckpts] == ["2", "10", "final"]
    assert ckpts[-1][1] == tmp_path / "adapter_model.safetensors"


def test_discover_checkpoints_mlx_layout_wins(tmp_path):
    (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-5" / "adapter_model.safetensors").write_bytes(b"")
    ckpts = discover_checkpoints(tmp_path)
    assert [label for label, _ in ckpts] == ["100", "final"]
```

Append to `tests/test_benchmark.py`:

```python
from autotune.benchmark import discover_adapter_dirs


def test_discover_adapter_dirs_peft(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    dirs = discover_adapter_dirs(tmp_path)
    assert dirs == [("10", tmp_path / "checkpoint-10"), ("final", tmp_path)]


def test_discover_adapter_dirs_mlx_stages_numbered(tmp_path):
    import json

    (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    dirs = discover_adapter_dirs(tmp_path)
    assert [label for label, _ in dirs] == ["100", "final"]
    staged = dict(dirs)["100"]
    assert (staged / "adapters.safetensors").is_symlink()
    assert dict(dirs)["final"] == tmp_path
```

Create `tests/test_train_sft.py`:

```python
"""Tests for train_sft's pure config builder (the mlx side; GPU drivers are smoke-tested)."""

from __future__ import annotations

from autotune.config import load_config
from autotune.train_sft import build_lora_config


def test_build_lora_config_save_every_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTUNE_BACKEND", "mlx")
    cfg = load_config()
    with_save = build_lora_config(cfg, tmp_path, tmp_path / "sft", iters=40, save_steps=10)
    assert with_save["save_every"] == 10
    without = build_lora_config(cfg, tmp_path, tmp_path / "sft", iters=40)
    assert "save_every" not in without
```

(If `load_config()` requires more env, read `tests/test_config.py` for the established pattern and mirror it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_weights_viz.py tests/test_benchmark.py tests/test_train_sft.py -q`
Expected: import/assertion failures.

- [ ] **Step 3: Implement**

`autotune/weights_viz.py` — replace `discover_checkpoints` body:

```python
_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")
_PEFT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def discover_checkpoints(adapter_dir: Path) -> list[tuple[str, Path]]:
    """Every checkpoint's safetensors under ``adapter_dir``, ordered, final last.

    Knows both training layouts: mlx (``NNNN_adapters.safetensors`` files + final
    ``adapters.safetensors``) and PEFT/cuda (``checkpoint-N/adapter_model.safetensors`` dirs +
    final ``adapter_model.safetensors``). mlx entries win when both exist.
    """
    numbered: list[tuple[int, Path]] = []
    for candidate in adapter_dir.glob("*_adapters.safetensors"):
        match = _CHECKPOINT_RE.match(candidate.name)
        if match is not None:
            numbered.append((int(match.group(1)), candidate))
    if not numbered:
        for candidate in adapter_dir.glob("checkpoint-*/adapter_model.safetensors"):
            match = _PEFT_DIR_RE.match(candidate.parent.name)
            if match is not None:
                numbered.append((int(match.group(1)), candidate))
    numbered.sort(key=lambda item: item[0])

    checkpoints: list[tuple[str, Path]] = [(str(step), path) for step, path in numbered]
    final = adapter_dir / "adapters.safetensors"
    peft_final = adapter_dir / "adapter_model.safetensors"
    if final.exists():
        checkpoints.append(("final", final))
    elif peft_final.exists():
        checkpoints.append(("final", peft_final))
    return checkpoints
```

`autotune/benchmark.py` — add after `stage_checkpoint`:

```python
def discover_adapter_dirs(adapter_dir: Path) -> list[tuple[str, Path]]:
    """Every checkpoint as ``(label, ready-to-load adapter dir)`` for ``make_proposer``.

    mlx numbered checkpoints are bare safetensors files, so they're staged via
    ``stage_checkpoint``; PEFT ``checkpoint-N/`` dirs already carry ``adapter_config.json`` +
    ``adapter_model.safetensors`` and load as-is; ``"final"`` is ``adapter_dir`` itself on both
    backends.
    """
    adapter_dir = Path(adapter_dir)
    dirs: list[tuple[str, Path]] = []
    for label, path in discover_checkpoints(adapter_dir):
        if label == "final":
            dirs.append((label, adapter_dir))
        elif path.name == "adapter_model.safetensors":
            dirs.append((label, path.parent))
        else:
            dirs.append((label, stage_checkpoint(adapter_dir, label, path)))
    return dirs
```

In `run_benchmark`, replace the `discover_checkpoints` + inner `stage_checkpoint` usage:

```python
    checkpoints = discover_adapter_dirs(adapter_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {adapter_dir} to benchmark.")
    ...
        for label, staged in checkpoints:
            proposer = make_proposer(cfg, staged)
```

(delete the now-unused `staged = stage_checkpoint(...)` line; keep everything else identical).

`autotune/train_sft.py`:

```python
def build_lora_config(
    cfg: Config, data_dir: Path, adapter_dir: Path, iters: int | None = None,
    save_steps: int | None = None,
) -> dict:
```

with, before `return`-assembly ends (add to the dict conditionally after building it):

```python
    config = { ...existing keys unchanged... }
    if save_steps is not None:
        config["save_every"] = save_steps
    return config
```

`_train_mlx(cfg, data_dir, adapter_dir, iters, save_steps=None)` passes it through to `build_lora_config`. `_train_cuda(cfg, data_dir, adapter_dir, iters, save_steps=None)` adds to `TRLSFTConfig(...)`:

```python
        **({"save_strategy": "steps", "save_steps": save_steps} if save_steps else {}),
```

`train(cfg, data_dir, iters=None, save_steps=None)` threads it to both. `main()` gains:

```python
    parser.add_argument(
        "--save-steps", type=int, default=None,
        help="checkpoint every N steps (numbered checkpoints for weights_viz/benchmark)",
    )
```

and passes `save_steps=args.save_steps`.

- [ ] **Step 4: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check autotune/ tests/
git add autotune/weights_viz.py autotune/benchmark.py autotune/train_sft.py tests/
git commit -m "feat(benchmark): PEFT/cuda checkpoint layout discovery + train_sft --save-steps"
```

---

### Task 6: `forest_benchmark.py` — per-checkpoint, per-domain forest sweep + plot

**Files:**
- Create: `autotune/forest_benchmark.py`
- Modify: `autotune/generate.py` (thread an optional system prompt — forest prompts must pair with `_FOREST_SYSTEM` for train/inference parity)
- Test: `tests/test_forest_benchmark.py` (new)

**Interfaces:**
- Consumes: `discover_adapter_dirs` (Task 5), `DOMAINS` / `domain_scores` / `score_forest` / `ForestVerdict` (Task 1), `follow_once` + `ROUTE_DEFAULT`, `_FOREST_SYSTEM` + `build_forest_mutation_prompt` (nudge_sft), `parse_genome_response` (nudge_steer), `resolve_states` (harvest), `base_genome`.
- Produces:
  - `generate.make_proposer(cfg, adapter_dir=None, system=None)`; `_generate_cuda(cfg, prompt, adapter_path, system=None)` and `_generate_mlx(...)` use `system or _SYSTEM`.
  - `forest_benchmark.propose_forest_genome(proposer, params, verdict) -> tuple[dict, bool]`
  - `forest_benchmark.ForestBenchRow`, `summarize_verdicts`, `format_forest_trend` (pure), `plot_forest_trend`, `run_forest_benchmark`, CLI.

- [ ] **Step 1: Write the failing tests** — `tests/test_forest_benchmark.py`:

```python
"""Tests for the per-domain forest checkpoint benchmark's pure seams."""

from __future__ import annotations

import json

from autotune.forest_benchmark import (
    ForestBenchRow,
    format_forest_trend,
    propose_forest_genome,
    summarize_verdicts,
)
from autotune.forest_story import score_forest


def _verdict(events):
    return score_forest(events)


_ENTER = {"event_type": "overworld", "turn": 0, "data": {"map_id": 51}}
_TWIN = {"event_type": "battle_outcome", "turn": 1, "data": {"battle_type": 2, "won": True}}
_EXIT = {"event_type": "overworld", "turn": 2, "data": {"map_id": 13}}


def test_propose_forest_genome_merges_parsed_over_params():
    def proposer(prompt):
        assert "Viridian Forest" in prompt
        return json.dumps({"hp_run_threshold": 0.15})

    genome, parsed = propose_forest_genome(proposer, {"hp_run_threshold": 0.6}, _verdict([_ENTER]))
    assert parsed and genome["hp_run_threshold"] == 0.15


def test_propose_forest_genome_falls_back_on_garbage():
    genome, parsed = propose_forest_genome(
        lambda _p: "not json at all", {"hp_run_threshold": 0.6}, _verdict([_ENTER])
    )
    assert not parsed and genome == {"hp_run_threshold": 0.6}


def test_summarize_verdicts_means_domains_across_states():
    row = summarize_verdicts("100", [_verdict([_ENTER, _TWIN, _EXIT]), _verdict([_ENTER])])
    assert row.label == "100"
    assert row.reward == 2.0  # (3 + 1) / 2
    assert row.domains == {"nav": 1.5, "battle": 0.5, "discovery": 0.0}
    assert row.crossed == 0.5


def test_format_forest_trend_has_domain_columns():
    baseline = ForestBenchRow(
        label="baseline", reward=2.0, domains={"nav": 1.0, "battle": 1.0, "discovery": 0.0},
        crossed=0.0, parsed=True,
    )
    row = ForestBenchRow(
        label="final", reward=4.0, domains={"nav": 2.0, "battle": 2.0, "discovery": 0.0},
        crossed=1.0, parsed=False,
    )
    table = format_forest_trend(baseline, [row])
    assert "nav" in table and "battle" in table and "discov" in table
    assert "baseline" in table and "final" in table
    assert "parse-fallback" in table  # unparsed rows are visibly flagged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_forest_benchmark.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Modify `autotune/generate.py`** — thread the system prompt:

```python
def _generate_mlx(cfg: Config, prompt: str, adapter_path: Path, system: str | None = None) -> str:
    ...
    from autotune.nudge_sft import _SYSTEM
    ...
    messages = [
        {"role": "system", "content": system or _SYSTEM},
        {"role": "user", "content": prompt},
    ]
```

(same one-line change in `_generate_cuda`), and:

```python
def make_proposer(
    cfg: Config, adapter_dir: Path | None = None, system: str | None = None
) -> Callable[[str], str]:
    """Return a ``(prompt) -> text`` proposer backed by the model + adapter (if trained).

    ``adapter_dir`` selects which trained adapter to load (default: the final adapter).
    ``system`` overrides the chat system prompt — the forest benchmark passes
    ``nudge_sft._FOREST_SYSTEM`` so inference pairs with what the forest examples trained.
    """
    adapter = adapter_dir if adapter_dir is not None else cfg.storage.adapter_dir

    if cfg.backend == "cuda":
        def _proposer(prompt: str) -> str:
            return _generate_cuda(cfg, prompt, adapter, system)
    else:
        def _proposer(prompt: str) -> str:
            return _generate_mlx(cfg, prompt, adapter, system)

    return _proposer
```

- [ ] **Step 4: Create `autotune/forest_benchmark.py`**

```python
"""Benchmark each LoRA checkpoint on the forest crossing, split into per-domain sub-scores.

The map-grained ``benchmark`` saturates flat at 6.0 on route1/first_battle — no gradient and no
per-domain signal. This sweep instead drives the genome-driven forest follower (navigation fixed
along the canonical route; the genome varies survival) once per checkpoint per forest state:
each checkpoint's proposer is asked to improve the base genome given a shared baseline crossing,
the proposed genome is run, and the dense ``forest_story`` reward is reported as nav / battle /
discovery sub-scores — the behavioural trend to pair 1:1 with ``weights_viz``'s weight-movement
trend. Item beats (2, 5, 7) score 0 until pokemon-kafka emits ``bag_count`` (accepted).

``propose_forest_genome`` / ``summarize_verdicts`` / ``format_forest_trend`` are pure and
unit-tested; the emulator sweep, plot, and CLI are exercised by the smoke run (AGENTS.md).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from autotune.benchmark import discover_adapter_dirs
from autotune.config import Config, load_config
from autotune.forest_follow import ROUTE_DEFAULT, follow_once
from autotune.forest_story import DOMAINS, ForestVerdict, domain_scores, score_forest
from autotune.generate import make_proposer
from autotune.genome import base_genome
from autotune.harvest import resolve_states
from autotune.nudge_sft import _FOREST_SYSTEM, build_forest_mutation_prompt
from autotune.nudge_steer import parse_genome_response


@dataclass(frozen=True)
class ForestBenchRow:
    """One checkpoint's mean forest performance across the benchmark states."""

    label: str
    reward: float  # mean beats_passed
    domains: dict[str, float]  # mean per-domain sub-score
    crossed: float  # fraction of states crossed
    parsed: bool  # False if ANY state's proposer output was unparseable (fallback genome ran)


def propose_forest_genome(
    proposer, params: dict, verdict: ForestVerdict
) -> tuple[dict, bool]:
    """Ask a checkpoint's proposer to improve ``params`` given the baseline ``verdict``.

    Returns ``(genome, parsed)``. Unparseable output falls back to ``params`` unchanged so the
    sweep still produces a row; the row is flagged instead of silently dropped.
    """
    text = proposer(build_forest_mutation_prompt(params, verdict))
    parsed = parse_genome_response(text)
    if parsed is None:
        return dict(params), False
    return {**params, **parsed}, True


def summarize_verdicts(
    label: str, verdicts: list[ForestVerdict], parsed: bool = True
) -> ForestBenchRow:
    """Mean the dense reward and per-domain sub-scores across states."""
    n = max(1, len(verdicts))
    return ForestBenchRow(
        label=label,
        reward=sum(v.reward for v in verdicts) / n,
        domains={d: sum(domain_scores(v)[d] for v in verdicts) / n for d in DOMAINS},
        crossed=sum(1 for v in verdicts if v.crossed) / n,
        parsed=parsed,
    )


def format_forest_trend(baseline: ForestBenchRow, rows: list[ForestBenchRow]) -> str:
    """Per-checkpoint table: total forest reward + nav/battle/discovery sub-scores."""
    lines = [
        "forest benchmark: proposer genome per checkpoint vs base genome "
        f"(baseline reward {baseline.reward:.2f})",
        f"  {'checkpoint':>10}  {'reward':>7}  {'nav':>5}  {'battle':>6}  {'discov':>6}  "
        f"{'crossed':>7}",
    ]
    for r in [baseline] + rows:
        flag = "" if r.parsed else "  (parse-fallback)"
        lines.append(
            f"  {r.label:>10}  {r.reward:>7.2f}  {r.domains['nav']:>5.2f}  "
            f"{r.domains['battle']:>6.2f}  {r.domains['discovery']:>6.2f}  "
            f"{r.crossed:>7.2f}{flag}"
        )
    return "\n".join(lines)


def plot_forest_trend(  # pragma: no cover - plotting, mirrors weights_viz.plot_norm_trends
    baseline: ForestBenchRow, rows: list[ForestBenchRow], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r.label for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, [r.reward for r in rows], marker="o", linewidth=2.5, label="total reward")
    for domain in DOMAINS:
        ax.plot(labels, [r.domains[domain] for r in rows], marker="o", label=domain)
    ax.axhline(baseline.reward, linestyle="--", color="gray", label="base-genome baseline")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("mean forest sub-beats reached")
    ax.set_title("Forest reward by checkpoint, split by behavior domain")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_forest_benchmark(  # pragma: no cover - emulator sweep, exercised by the smoke run
    cfg: Config,
    adapter_dir: Path,
    state_paths: list[str],
    max_steps: int,
    worldmap_in: str | None,
) -> tuple[ForestBenchRow, list[ForestBenchRow]]:
    """Sweep every checkpoint's proposed genome through the forest follower, per state.

    The base-genome baseline run is adapter-independent, so it runs once per state and doubles
    as the situation the proposer is asked to improve.
    """
    checkpoints = discover_adapter_dirs(Path(adapter_dir))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {adapter_dir} to benchmark.")
    if cfg.env.rom_path is None:
        raise RuntimeError("ROM_PATH is not set — point it at a Pokemon Red ROM.")
    rom = str(cfg.env.rom_path.resolve())
    route = ROUTE_DEFAULT if Path(ROUTE_DEFAULT).exists() else None

    baseline_verdicts: list[ForestVerdict] = []
    per_ckpt: dict[str, list[ForestVerdict]] = {label: [] for label, _ in checkpoints}
    parsed_ok: dict[str, bool] = {label: True for label, _ in checkpoints}

    for state in state_paths:
        base_result = follow_once(
            rom, state, base_genome(), max_steps=max_steps, worldmap_in=worldmap_in, route=route
        )
        base_verdict = score_forest(base_result["events"])
        baseline_verdicts.append(base_verdict)
        print(f"[forest-bench] {Path(state).name}: baseline reward={base_verdict.reward}")

        for label, ckpt_dir in checkpoints:
            proposer = make_proposer(cfg, ckpt_dir, system=_FOREST_SYSTEM)
            genome, parsed = propose_forest_genome(proposer, base_genome(), base_verdict)
            parsed_ok[label] = parsed_ok[label] and parsed
            result = follow_once(
                rom, state, genome, max_steps=max_steps, worldmap_in=worldmap_in, route=route
            )
            verdict = score_forest(result["events"])
            per_ckpt[label].append(verdict)
            print(
                f"[forest-bench] {Path(state).name}: checkpoint {label} "
                f"reward={verdict.reward} domains={domain_scores(verdict)} parsed={parsed}"
            )

    baseline = summarize_verdicts("baseline", baseline_verdicts)
    rows = [summarize_verdicts(label, per_ckpt[label], parsed_ok[label]) for label, _ in checkpoints]
    return baseline, rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Benchmark each LoRA checkpoint on the forest crossing, per behavior domain."
    )
    p.add_argument("--adapter-dir", type=Path, default=Path("out/sft"))
    p.add_argument("--states", default="states/forest_lv", help="a .state file or a dir of them")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--worldmap-in", default="states/forest_follow_wm.json")
    p.add_argument("--out", type=Path, default=Path("out/forest_benchmark_trends.png"))
    args = p.parse_args(argv)

    cfg = load_config()
    state_paths = resolve_states(args.states)
    if not state_paths:
        print(f"No save states found at {args.states!r}.", file=sys.stderr)
        return 2
    wm = args.worldmap_in if args.worldmap_in and Path(args.worldmap_in).exists() else None

    baseline, rows = run_forest_benchmark(
        cfg, args.adapter_dir, state_paths, args.max_steps, wm
    )
    print(format_forest_trend(baseline, rows))
    plot_forest_trend(baseline, rows, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 5: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check autotune/ tests/
git add autotune/forest_benchmark.py autotune/generate.py tests/test_forest_benchmark.py
git commit -m "feat(forest_benchmark): per-checkpoint forest sweep with per-domain trend table and plot"
```

---

### Task 7: End-to-end smoke — harvest ×2 → merge → train_sft → forest_benchmark

Execution task, no new code. Backend on this machine is cuda (Linux + RTX 5090); the issue wrote "mlx" from a Mac — `resolve_backend()` picks the right one. All emulator runs need `.env` (`POKEMON_KAFKA_DIR`, `ROM_PATH`) which exists in this checkout. Run each command, capture the tail of its output, and stop at the first failure (report it — don't improvise fixes beyond obvious CLI-arg mistakes).

- [ ] **Step 1: Confirm GPU + env**

Run: `uv run python smoke_cuda.py` — expect a CUDA-available confirmation.

- [ ] **Step 2: Forest harvest (battle/discovery pairs)**

Run: `uv run python -m autotune.forest_harvest --run-thresholds 0.1,0.35,0.6 --heal-thresholds 0.25 --max-steps 900 --out-dir out/forest_sft 2>&1 | tail -20`
Expected: `[harvest] N SFT pairs ...`, `out/forest_sft/corpus.jsonl` exists with a `domains` key on every row. If it reports "no gradient", re-run with `--max-steps 1500` and thresholds `0.05,0.3,0.7`.

- [ ] **Step 3: Map-grained harvest (nav pairs)**

Run: `uv run python -m autotune.harvest --states states/route2.state --genomes 6 --max-turns 400 2>&1 | tail -10`
Expected: `data/sft/corpus.jsonl` exists, rows tagged `["nav"]`. If route2.state yields 0 examples, try `--states states/viridian_city.state --genomes 8`.

- [ ] **Step 4: Merge**

Run: `uv run python -m autotune.merge_corpus --inputs out/forest_sft/corpus.jsonl data/sft/corpus.jsonl --out-dir data/sft_union`
Expected: census line naming at least `nav` and one of `battle`/`discovery`; `data/sft_union/{corpus,train,valid}.jsonl` written; train rows have no `domains` key.

- [ ] **Step 5: Train with checkpoints**

Run: `uv run python -m autotune.train_sft --data-dir data/sft_union --iters 40 --save-steps 10 2>&1 | tail -5`
Expected: adapter in `out/sft` with numbered checkpoints (`out/sft/checkpoint-10` etc. on cuda).

- [ ] **Step 6: Benchmark + weights pairing**

Run: `uv run python -m autotune.forest_benchmark --adapter-dir out/sft --states states/forest_lv/lead_lv13_potions.state --max-steps 900 --out out/forest_benchmark_trends.png`
Expected: per-domain trend table (baseline + one row per checkpoint), `out/forest_benchmark_trends.png` written, exit 0.

Run: `uv run python -m autotune.weights_viz --adapter-dir out/sft --out out/lora_weight_trends.png`
Expected: `Wrote out/lora_weight_trends.png` (PEFT layout now discovered).

- [ ] **Step 7: Record + commit smoke evidence**

Append the trend table and census to `docs/experiment-findings.md` under a `## 2026-07-02 domain-tagged corpus smoke` heading (2–10 lines, verbatim output). Commit:

```bash
git add docs/experiment-findings.md
git commit -m "docs: record domain-tagged corpus + per-domain benchmark smoke results"
```

---

## Self-Review Notes

- Issue scope items 1–6 map to Tasks 1, 2, 3, 4, (5+6), 7 respectively; Task 5 exists because the issue's benchmark must run on this cuda box and short smoke runs need `--save-steps` to produce numbered checkpoints at all.
- `build_forest_pair_example` signature change is contained: `build_forest_dataset` is its only production caller.
- `write_sft_data` stripping is backward-compatible with untagged in-loop examples (`loop.py`) — `_strip_tags` only keeps `messages`, which every example has. After Task 3, loop-built map examples are tagged nav by default, which is correct and stripped before training.
- Types line up: `pair_domains` returns `tuple[str, ...]`, stored as `list` in JSON rows; `domain_scores` returns `dict[str, int]`; `ForestBenchRow.domains` is `dict[str, float]` (means).
