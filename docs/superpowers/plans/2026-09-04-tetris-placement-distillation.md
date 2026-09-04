# Tetris Placement Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline that turns `gemma4:26b` Tetris games into a corpus, fine-tunes `google/gemma-4-E4B-it` on it with HF Jobs, packages the result back into Ollama, and gates it on the live benchmark over held-out seeds.

**Architecture:** Tetris gains one optional `board` field on its `placement_graded` event. Everything else lives in `empirical-evidence`: `tetris_rollout.py` mints games under a tapes proxy, `tetris_corpus.py` turns `runs/` into records and training views (rendering prompts with tetris's own `build_user_prompt`), two self-contained uv scripts train and package on HF Jobs, and `tetris_eval.py` applies the three-rule gate to a `tetris-bench` result file. HF Jobs scripts depend only on PyPI and the Hub — never on a git checkout — so their pure functions are unit-tested locally and their GPU paths are smoke-tested.

**Tech Stack:** Python 3.11+, uv, numpy, TRL/PEFT/transformers (in the job only), `hf` CLI, Ollama, `tapes serve proxy`, llama.cpp (in the packaging job only), pytest + ruff (line length 100, `E F I W`).

**Spec:** `docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md` — read it first; this plan argues from it.

## Global Constraints

- Tetris: no new file, flag, module or seam. One optional field on `placement_graded`; `build_graded_event(turn, grade)` with no board stays byte-identical to today.
- Seed split: eval seeds are `{1, 2, 3, 4, 5}`; training seeds are `>= 100`. Seed 1 is never trained on. The corpus builder refuses rows on the wrong side.
- Filters: drop the last **5** decisions of any topped-out run; veto a decision whose `chosen_value <= TOP_OUT_VALUE` while `best_value > TOP_OUT_VALUE`. Nothing else filters on the oracle.
- SFT user turn is rendered by `tetris_agent.prompts.build_user_prompt` with `deadline_s=15.0`, plus the exact `PI_JSON_INSTRUCTIONS` / `PI_PROMPT_SUFFIX` the pi arm appends — a byte comparison in tests, not a resemblance.
- Teacher arm: `pi/gemma4:26b`, `features`, effort `off`, `--fixed-effort`, `--paused`, `--max-pieces 100`. Student: `google/gemma-4-E4B-it`, served as `gemma4:latest` at Q4_K_M. Tuned tag is served at **Q4_K_M** too.
- Gate: median race over seeds 1–5 beats the baseline's median; tokens per decision `<= 1.5 ×` baseline; late decisions do not increase. All three.
- Every tetris change goes on a branch from `main` in `../tetris`; every empirical-evidence change goes on `feat/tetris-placement-distillation`.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL`.

---

## File Structure

**tetris** (`../tetris`, branch `feat/graded-board`)
- Modify `src/tetris_agent/events.py` — `build_graded_event(turn, grade, board=None)`, `EventCollector.graded(grade, turn=None, board=None)`.
- Modify `src/tetris_agent/agent.py:166`, `src/tetris_agent/live_agent.py:119` — pass the board.
- Modify `tests/test_events.py`, `tests/test_agent.py`, `tests/test_live_agent.py`.
- Modify `src/tetris_agent/pricing.py` (Task 11 only) — one `MODELS` line for the tuned tag.

**empirical-evidence** (branch `feat/tetris-placement-distillation`)
- Modify `pyproject.toml` — `tetris` extra sourcing `../tetris`; coverage omits for the subprocess/GPU modules.
- Create `autotune/tetris_corpus.py` — `read_run`, filters, split, SFT view, eval view, CLI.
- Create `autotune/tetris_rollout.py` — proxy lifecycle, preflight, `tetris-bench` invocation, manifest.
- Create `autotune/tetris_train_job.py` — self-contained uv script for HF Jobs: LoRA SFT + tier-1 eval + upload. Pure functions `parse_placement`, `grade_answers` tested locally.
- Create `autotune/tetris_package_job.py` — self-contained uv script for HF Jobs: merge, GGUF, Q4_K_M, upload.
- Create `autotune/tetris_eval.py` — the tier-2 gate over a `tetris-bench` result file, plus CLI.
- Create `scripts/register_pi_tag.py` — adds an Ollama tag to `~/.pi/agent/models.json` with a backup.
- Create `tests/test_tetris_corpus.py`, `tests/test_tetris_rollout.py`, `tests/test_tetris_train_job.py`, `tests/test_tetris_eval.py`, `tests/fixtures/tetris_run/` (one synthetic run dir).
- Modify `README.md` — a "Tetris" section; the spec (two sentences, Task 13).

---

## Phase A — tetris

### Task 1: The board rides on `placement_graded`

**Files:**
- Modify: `src/tetris_agent/events.py:56-65` (`build_graded_event`), `:110-111` (`EventCollector.graded`)
- Modify: `src/tetris_agent/agent.py:166`
- Modify: `src/tetris_agent/live_agent.py:119`
- Test: `tests/test_events.py`, `tests/test_agent.py`, `tests/test_live_agent.py`

**Interfaces:**
- Produces: `build_graded_event(turn: int, grade, board: np.ndarray | None = None) -> dict` — when `board` is given, `event["data"]["board"]` is a list of 18 strings of 10 chars, `.` empty / `#` settled, row 0 first. `EventCollector.graded(grade, turn=None, board=None)`.

- [ ] **Step 1: Branch from main in the tetris checkout**

```bash
cd ../tetris && git checkout main && git pull && git checkout -b feat/graded-board
```

- [ ] **Step 2: Write the failing event tests**

Append to `tests/test_events.py`:

```python
def test_graded_event_carries_the_board_when_given():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import build_graded_event
    from tetris_agent.policy import Placement

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    g = quality.grade(board, "O", "I", Placement(rotation=0, col=7, score=0.0))

    event = build_graded_event(turn=4, grade=g, board=board)
    rows = event["data"]["board"]
    assert len(rows) == 18 and all(len(r) == 10 for r in rows)
    assert rows[0] == ".........."
    assert rows[15] == "######...."
    # Round-trips through the encoding situation_corpus already reads.
    back = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    assert (back == board).all()


def test_graded_event_without_a_board_is_unchanged():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import build_graded_event
    from tetris_agent.policy import Placement

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    g = quality.grade(board, "O", "I", Placement(rotation=0, col=7, score=0.0))
    event = build_graded_event(turn=4, grade=g)
    assert "board" not in event["data"]
    assert set(event["data"]) == set(g.to_dict())


def test_collector_passes_the_board_through():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Placement

    published = []

    class Sink:
        def publish(self, event):
            published.append(event)

    board = np.zeros((18, 10), dtype=bool)
    board[17, :] = True
    board[17, 3] = False
    collector = EventCollector(Sink())
    collector.graded(quality.grade(board, "O", "I", Placement(rotation=0, col=0, score=0.0)), board=board)
    assert published[0]["data"]["board"][17] == "###.######"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cd ../tetris && uv run pytest tests/test_events.py -k "board" -v`
Expected: 3 FAIL — `TypeError: build_graded_event() got an unexpected keyword argument 'board'` and the same for `graded()`.

- [ ] **Step 4: Implement in `events.py`**

Replace `build_graded_event` and `EventCollector.graded`:

```python
def build_graded_event(turn: int, grade, board=None) -> dict:
    """A decision scored against the lookahead oracle (see quality.py).

    Separate from `placement_decision` because that event is published before
    execution, so the viewer can render the think-freeze, and the grade does not
    exist until the piece has locked.

    `board` is the pre-decision board the grade was computed on, encoded the way
    situation_corpus.py reads it (`.` empty, `#` settled, row 0 first). Optional,
    so an event built without it is byte-identical to before the field existed.
    """
    data = grade.to_dict()
    if board is not None:
        data["board"] = ["".join("#" if cell else "." for cell in row) for row in board]
    return _envelope("placement_graded", turn, data)
```

```python
    def graded(self, grade, turn: int | None = None, board=None) -> None:
        self.publisher.publish(build_graded_event(self.turn if turn is None else turn, grade, board=board))
```

- [ ] **Step 5: Run the event tests to verify they pass**

Run: `cd ../tetris && uv run pytest tests/test_events.py -v`
Expected: all PASS, including the pre-existing graded tests.

- [ ] **Step 6: Write the failing agent tests**

Append to `tests/test_agent.py` (uses the module's existing `_paused_agent_fixtures` and `_fake_grader`):

```python
def test_the_graded_event_carries_the_pre_decision_board(monkeypatch):
    calls = []
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader(calls))
    agent.run(timer_div=0)
    graded = next(e for e in publisher.events if e["event_type"] == "placement_graded")
    rows = graded["data"]["board"]
    assert len(rows) == 18 and all(len(r) == 10 and set(r) <= {".", "#"} for r in rows)
    expected = ["".join("#" if c else "." for c in row) for row in calls[0][0]]
    assert rows == expected
```

Append to `tests/test_live_agent.py` (uses `LiveFakeEmulator`, `install_live`, `install_controller`, `CapturingPublisher`, `make_agent`, `ScriptedPolicy`, `ManualThreads`, `_fake_grader`, `state_with`, `falling_at` exactly as `test_live_grades_an_executed_decision_after_the_lock` does):

```python
def test_live_graded_event_carries_the_board_the_grader_saw(monkeypatch):
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    calls = []
    agent = make_agent(
        emu, ScriptedPolicy([Placement(0, 0, 0.0)]), pub, ManualThreads(immediate=True),
        max_pieces=1, grader=_fake_grader(calls),
    )
    agent.run(timer_div=0)
    graded = pub.of_type("placement_graded")[0]
    expected = ["".join("#" if c else "." for c in row) for row in calls[0][0]]
    assert graded["data"]["board"] == expected
```

- [ ] **Step 7: Run them to verify they fail**

Run: `cd ../tetris && uv run pytest tests/test_agent.py tests/test_live_agent.py -k "carries_the" -v`
Expected: 2 FAIL with `KeyError: 'board'`.

- [ ] **Step 8: Pass the board at both call sites**

`src/tetris_agent/agent.py:166` — change `self.collector.graded(g)` to:

```python
                    self.collector.graded(g, board=state.board)
```

`src/tetris_agent/live_agent.py:119` — change `self.collector.graded(g)` to:

```python
                        self.collector.graded(g, board=board)
```

- [ ] **Step 9: Run the full tetris suite and ruff**

Run: `cd ../tetris && uv run pytest -q && uv run ruff check`
Expected: all PASS; ruff clean.

- [ ] **Step 10: Commit**

```bash
cd ../tetris && git add src/tetris_agent/events.py src/tetris_agent/agent.py src/tetris_agent/live_agent.py tests/test_events.py tests/test_agent.py tests/test_live_agent.py
git commit -F - <<'EOF'
Graded events carry the pre-decision board

One optional field on placement_graded, passed from the two grade call sites
that already hold it. An event built without a board is byte-identical to
before. This is the whole tetris side of the placement-distillation design
(empirical-evidence/docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md):
the corpus needs the board, and everything else is derivable from it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

---

## Phase B — empirical-evidence: data

### Task 2: The tetris dependency and a synthetic run fixture

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/fixtures/tetris_run/20260904-000000-abcdef/events.jsonl`, `.../summary.json`, `.../meta.json`

**Interfaces:**
- Produces: `import tetris_agent` works under `uv run` after `uv sync --extra tetris`; a fixture run dir with 4 spawns, 4 decisions (one late), 3 graded events (one without a board), and a topped-out summary.

- [ ] **Step 1: Add the extra and coverage omits**

In `pyproject.toml`, add under `[project.optional-dependencies]`:

```toml
# The tetris benchmark, imported for its prompt renderer and grader so training
# inputs are byte-identical to inference inputs. Sibling checkout, like pokemon-kafka.
tetris = [
    "tetris-agent",
]
```

Add to `[tool.uv.sources]`:

```toml
tetris-agent = { path = "../tetris", editable = true }
```

Extend `[tool.coverage.run] omit` with:

```toml
    "autotune/tetris_rollout.py",
    "autotune/tetris_package_job.py",
```

- [ ] **Step 2: Sync and prove the import**

Run: `uv sync --extra tetris && uv run python -c "from tetris_agent.prompts import build_user_prompt; from tetris_agent.quality import rank_placements; print('ok')"`
Expected: `ok`. (Tetris pins `pyboy>=2.4`, this repo pins `pyboy==2.7.0`; they resolve.)

- [ ] **Step 3: Write the fixture run**

`tests/fixtures/tetris_run/20260904-000000-abcdef/meta.json`:

```json
{"label": "pi/gemma4:26b/features/off+fixed", "recorded_at": "2026-09-04T00:00:00+00:00"}
```

`tests/fixtures/tetris_run/20260904-000000-abcdef/summary.json`:

```json
{"run_id": "20260904-000000-abcdef", "fitness": {"score": 120, "lines": 3, "level": 0, "pieces_placed": 4, "avg_holes": 0.5, "max_stack_height": 6, "misexec_count": 0, "max_misexec_streak": 0, "topped_out": true, "graded_decisions": 3, "mean_regret": 0.1, "top1_rate": 0.667, "top3_rate": 1.0, "policy": {"policy": "pi/gemma4:26b/features/off", "model": "pi/gemma4:26b", "harness": "features", "effort": "off", "decisions": 4, "late": 1}}, "params": {}}
```

`tests/fixtures/tetris_run/20260904-000000-abcdef/events.jsonl` — one JSON object per line. `E` is the empty row `".........."` and `F` is the floor row `"##########"` written out in full; the board is 18 rows:

```json
{"schema": "tetris.game.v1", "event_type": "session", "turn": 0, "occurred_at": "2026-09-04T00:00:00+00:00", "data": {"phase": "start", "policy": "pi/gemma4:26b/features/off", "arm": "pi/gemma4:26b/features/off+fixed", "model": "pi/gemma4:26b", "harness": "features", "effort": "off", "mode": "paused", "seed": 100, "max_pieces": 4}}
{"schema": "tetris.game.v1", "event_type": "piece_spawn", "turn": 1, "occurred_at": "2026-09-04T00:00:01+00:00", "data": {"piece": "I", "next_piece": "O"}}
{"schema": "tetris.game.v1", "event_type": "placement_decision", "turn": 1, "occurred_at": "2026-09-04T00:00:02+00:00", "data": {"rotation": 0, "col": 3, "score": 0.0, "reason": "Flat on the floor.", "latency_ms": 2100.0, "tokens": 30}}
{"schema": "tetris.game.v1", "event_type": "piece_locked", "turn": 1, "occurred_at": "2026-09-04T00:00:03+00:00", "data": {"lines_delta": 0, "misexec": 0, "score": 0, "holes": 0, "agg_height": 4, "bumpiness": 2, "max_height": 1}}
{"schema": "tetris.game.v1", "event_type": "placement_graded", "turn": 1, "occurred_at": "2026-09-04T00:00:03+00:00", "data": {"chosen": [0, 3], "best": [0, 3], "chosen_value": -2.0, "best_value": -2.0, "worst_value": -6.0, "rank": 1, "legal_count": 17, "regret": 0.0, "regret_norm": 0.0, "ply": 2, "genome": {"w_lines": 0.760666, "w_agg_height": -0.510066, "w_holes": -0.35663, "w_bumpiness": -0.184483}, "board": ["..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".........."]}}
{"schema": "tetris.game.v1", "event_type": "piece_spawn", "turn": 2, "occurred_at": "2026-09-04T00:00:04+00:00", "data": {"piece": "O", "next_piece": "T"}}
{"schema": "tetris.game.v1", "event_type": "placement_decision", "turn": 2, "occurred_at": "2026-09-04T00:00:05+00:00", "data": {"rotation": 0, "col": 0, "score": 0.0, "reason": "Left corner.", "latency_ms": 1900.0, "tokens": 28}}
{"schema": "tetris.game.v1", "event_type": "piece_locked", "turn": 2, "occurred_at": "2026-09-04T00:00:06+00:00", "data": {"lines_delta": 0, "misexec": 0, "score": 0, "holes": 0, "agg_height": 8, "bumpiness": 4, "max_height": 2}}
{"schema": "tetris.game.v1", "event_type": "placement_graded", "turn": 2, "occurred_at": "2026-09-04T00:00:06+00:00", "data": {"chosen": [0, 0], "best": [0, 7], "chosen_value": -4.0, "best_value": -3.0, "worst_value": -1000000.0, "rank": 2, "legal_count": 9, "regret": 1.0, "regret_norm": 0.000001, "ply": 2, "genome": {"w_lines": 0.760666, "w_agg_height": -0.510066, "w_holes": -0.35663, "w_bumpiness": -0.184483}, "board": ["..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "...####..."]}}
{"schema": "tetris.game.v1", "event_type": "piece_spawn", "turn": 3, "occurred_at": "2026-09-04T00:00:07+00:00", "data": {"piece": "T", "next_piece": "S"}}
{"schema": "tetris.game.v1", "event_type": "placement_decision", "turn": 3, "occurred_at": "2026-09-04T00:00:08+00:00", "data": {"rotation": 2, "col": 5, "score": 0.0, "reason": "Fills the gap.", "latency_ms": 2000.0, "tokens": 31}}
{"schema": "tetris.game.v1", "event_type": "piece_locked", "turn": 3, "occurred_at": "2026-09-04T00:00:09+00:00", "data": {"lines_delta": 0, "misexec": 0, "score": 0, "holes": 0, "agg_height": 11, "bumpiness": 5, "max_height": 2}}
{"schema": "tetris.game.v1", "event_type": "placement_graded", "turn": 3, "occurred_at": "2026-09-04T00:00:09+00:00", "data": {"chosen": [2, 5], "best": [2, 5], "chosen_value": -1000000.0, "best_value": -3.5, "worst_value": -1000000.0, "rank": 9, "legal_count": 17, "regret": 999996.5, "regret_norm": 1.0, "ply": 2, "genome": {"w_lines": 0.760666, "w_agg_height": -0.510066, "w_holes": -0.35663, "w_bumpiness": -0.184483}, "board": ["..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "##........", "#######..."]}}
{"schema": "tetris.game.v1", "event_type": "piece_spawn", "turn": 4, "occurred_at": "2026-09-04T00:00:10+00:00", "data": {"piece": "S", "next_piece": "Z"}}
{"schema": "tetris.game.v1", "event_type": "placement_decision", "turn": 4, "occurred_at": "2026-09-04T00:00:30+00:00", "data": {"rotation": 0, "col": 2, "score": 0.0, "reason": "Too slow.", "latency_ms": 19000.0, "late": true}}
{"schema": "tetris.game.v1", "event_type": "piece_locked", "turn": 4, "occurred_at": "2026-09-04T00:00:31+00:00", "data": {"lines_delta": 0, "misexec": 0, "score": 0, "holes": 2, "agg_height": 30, "bumpiness": 9, "max_height": 18}}
{"schema": "tetris.game.v1", "event_type": "game_over", "turn": 4, "occurred_at": "2026-09-04T00:00:32+00:00", "data": {"fitness": {"score": 120, "pieces_placed": 4, "topped_out": true}}}
```

Note what the fixture encodes on purpose: turn 3's `chosen_value` is `TOP_OUT_VALUE` with a survivable `best_value` (the veto case); turn 4 is late and ungraded; the run topped out.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/fixtures/tetris_run
git commit -F - <<'EOF'
tetris extra and a synthetic run fixture

tetris-agent is a sibling-path dependency behind `--extra tetris`, imported for
build_user_prompt and rank_placements so the corpus renders prompts with the
same code that serves them. The fixture run carries one veto case, one late
decision, and a top-out, for the corpus tests to come.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 3: `read_run` — records and counted exclusions

**Files:**
- Create: `autotune/tetris_corpus.py`
- Test: `tests/test_tetris_corpus.py`

**Interfaces:**
- Produces: `Record` dataclass; `read_run(run_dir: Path) -> tuple[list[Record], Counter]`. `Record` fields: `run_id, turn, arm, model, harness, effort, seed, mode, board (list[str]), piece, next_piece, chosen (list[int]), reason, grade (dict), outcome (dict)`. `outcome` has `final_score, lines, pieces_placed, topped_out, pieces_after`. Exclusion keys: `late`, `ungraded`, `no_board`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tetris_corpus.py`:

```python
from pathlib import Path

from autotune.tetris_corpus import read_run

FIXTURE = Path(__file__).parent / "fixtures" / "tetris_run" / "20260904-000000-abcdef"


def test_read_run_yields_one_record_per_graded_event_with_a_board():
    records, exclusions = read_run(FIXTURE)
    assert [r.turn for r in records] == [1, 2, 3]
    assert exclusions == {"late": 1}


def test_read_run_fills_state_choice_and_outcome():
    records, _ = read_run(FIXTURE)
    r = records[1]
    assert r.run_id == "20260904-000000-abcdef"
    assert (r.arm, r.model, r.harness, r.effort, r.seed, r.mode) == (
        "pi/gemma4:26b/features/off+fixed", "pi/gemma4:26b", "features", "off", 100, "paused"
    )
    assert (r.piece, r.next_piece, r.chosen, r.reason) == ("O", "T", [0, 0], "Left corner.")
    assert len(r.board) == 18 and r.board[17] == "...####..."
    assert r.grade["rank"] == 2 and r.grade["genome"]["w_lines"] == 0.760666
    assert r.outcome == {
        "final_score": 120, "lines": 3, "pieces_placed": 4, "topped_out": True, "pieces_after": 2,
    }


def test_read_run_skips_graded_events_without_a_board(tmp_path):
    import json
    import shutil

    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    lines = (run / "events.jsonl").read_text().splitlines()
    out = []
    for line in lines:
        e = json.loads(line)
        if e["event_type"] == "placement_graded" and e["turn"] == 2:
            del e["data"]["board"]
        out.append(json.dumps(e))
    (run / "events.jsonl").write_text("\n".join(out) + "\n")
    records, exclusions = read_run(run)
    assert [r.turn for r in records] == [1, 3]
    assert exclusions == {"late": 1, "no_board": 1}


def test_read_run_parses_a_live_session_without_arm_fields(tmp_path):
    import json
    import shutil

    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    lines = (run / "events.jsonl").read_text().splitlines()
    session = json.loads(lines[0])
    session["data"] = {"phase": "start", "policy": "pi/gemma4:26b/features/off", "mode": "live", "level": 0, "timer_div": 7}
    lines[0] = json.dumps(session)
    (run / "events.jsonl").write_text("\n".join(lines) + "\n")
    records, _ = read_run(run)
    r = records[0]
    assert (r.model, r.harness, r.effort, r.seed, r.mode) == ("pi/gemma4:26b", "features", "off", 7, "live")
    assert r.arm == "pi/gemma4:26b/features/off+fixed"  # from meta.json when the session has none
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_corpus.py -v`
Expected: `ModuleNotFoundError: No module named 'autotune.tetris_corpus'`.

- [ ] **Step 3: Implement `read_run`**

`autotune/tetris_corpus.py`:

```python
"""Tetris placement corpus: runs/ -> neutral records -> SFT and eval views.

One row per graded decision. The board comes off the `placement_graded` event;
everything derivable from a board (the full ranking, a deeper grade) is
recomputed here, never stored. Prompts are rendered by tetris's own
`build_user_prompt` so training input is byte-identical to inference input.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

GRADE_KEYS = (
    "rank", "legal_count", "regret", "regret_norm", "best",
    "chosen_value", "best_value", "worst_value", "genome", "ply",
)


@dataclass(frozen=True)
class Record:
    run_id: str
    turn: int
    arm: str
    model: str
    harness: str | None
    effort: str | None
    seed: int
    mode: str
    board: list[str]
    piece: str
    next_piece: str
    chosen: list[int]
    reason: str
    grade: dict
    outcome: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_policy(policy: str) -> tuple[str, str | None, str | None]:
    """`pi/gemma4:26b/features/off` -> (model, harness, effort); `heuristic` -> (heuristic, None, None)."""
    parts = policy.split("/")
    if parts[0] == "pi" and len(parts) >= 2:
        model, rest = "/".join(parts[:2]), parts[2:]
    else:
        model, rest = parts[0], parts[1:]
    harness = rest[0] if rest else None
    effort = rest[1] if len(rest) > 1 else None
    return model, harness, effort


def read_run(run_dir: Path) -> tuple[list[Record], Counter]:
    """Records for every graded decision that carries a board, plus counted exclusions."""
    run_dir = Path(run_dir)
    events = _events(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    fitness = summary.get("fitness", {})

    session = next(
        (e["data"] for e in events if e.get("event_type") == "session" and e.get("data", {}).get("phase") == "start"),
        {},
    )
    policy = session.get("policy", "")
    p_model, p_harness, p_effort = _parse_policy(policy)
    model = session.get("model") or p_model
    harness = session.get("harness") or p_harness
    effort = session.get("effort") if "effort" in session else p_effort
    arm = session.get("arm") or meta.get("label") or policy
    seed = int(session.get("seed", session.get("timer_div", 0)))
    mode = session.get("mode", "paused")

    spawns = {e["turn"]: e["data"] for e in events if e.get("event_type") == "piece_spawn"}
    decisions: dict[int, dict] = {}
    for e in events:
        if e.get("event_type") == "placement_decision":
            decisions[e["turn"]] = e["data"]  # the last decision for a turn is the accepted one
    graded = {e["turn"]: e["data"] for e in events if e.get("event_type") == "placement_graded"}

    exclusions: Counter = Counter()
    for turn, d in decisions.items():
        if turn not in graded:
            exclusions["late" if d.get("late") else "ungraded"] += 1

    pieces_placed = int(fitness.get("pieces_placed", 0))
    outcome_base = {
        "final_score": int(fitness.get("score", 0)),
        "lines": int(fitness.get("lines", 0)),
        "pieces_placed": pieces_placed,
        "topped_out": bool(fitness.get("topped_out", False)),
    }

    records: list[Record] = []
    for turn in sorted(graded):
        g = graded[turn]
        if "board" not in g:
            exclusions["no_board"] += 1
            continue
        spawn = spawns.get(turn, {})
        records.append(
            Record(
                run_id=summary.get("run_id", run_dir.name),
                turn=turn,
                arm=arm,
                model=model,
                harness=harness,
                effort=effort,
                seed=seed,
                mode=mode,
                board=list(g["board"]),
                piece=spawn.get("piece", ""),
                next_piece=spawn.get("next_piece", ""),
                chosen=list(g["chosen"]),
                reason=decisions.get(turn, {}).get("reason", ""),
                grade={k: g[k] for k in GRADE_KEYS if k in g},
                outcome={**outcome_base, "pieces_after": pieces_placed - turn},
            )
        )
    return records, exclusions
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_corpus.py -v && uv run ruff check autotune/tetris_corpus.py`
Expected: 4 PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add autotune/tetris_corpus.py tests/test_tetris_corpus.py
git commit -F - <<'EOF'
tetris_corpus: read a run into graded records

One Record per placement_graded event that carries a board; late, ungraded
and board-less decisions are counted, never replayed. Live sessions, which
carry no arm fields, are parsed from the policy string and meta.json.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 4: Filters and the seed split

**Files:**
- Modify: `autotune/tetris_corpus.py`
- Test: `tests/test_tetris_corpus.py`

**Interfaces:**
- Produces: `apply_filters(records, death_spiral=5) -> tuple[list[Record], Counter]` (keys `death_spiral`, `top_out_veto`); `split(records) -> tuple[list[Record], list[Record], Counter]` returning `(train, valid, exclusions)` with key `seed_out_of_pool`; constants `TOP_OUT_VALUE`, `DEATH_SPIRAL = 5`, `TRAIN_SEED_MIN = 100`, `EVAL_SEEDS = (1, 2, 3, 4, 5)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tetris_corpus.py`:

```python
from dataclasses import replace

from autotune.tetris_corpus import EVAL_SEEDS, TRAIN_SEED_MIN, apply_filters, split


def _records(n, *, run_id="r", seed=100, topped_out=False, chosen_value=-2.0, best_value=-1.0):
    base, _ = read_run(FIXTURE)
    proto = base[0]
    out = []
    for turn in range(1, n + 1):
        out.append(
            replace(
                proto,
                run_id=run_id,
                turn=turn,
                seed=seed,
                grade={**proto.grade, "chosen_value": chosen_value, "best_value": best_value},
                outcome={**proto.outcome, "pieces_placed": n, "topped_out": topped_out, "pieces_after": n - turn},
            )
        )
    return out


def test_death_spiral_drops_exactly_the_last_five_of_a_topped_out_run():
    kept, dropped = apply_filters(_records(12, topped_out=True))
    assert [r.turn for r in kept] == list(range(1, 8))
    assert dropped == {"death_spiral": 5}


def test_a_survived_run_keeps_every_decision():
    kept, dropped = apply_filters(_records(12, topped_out=False))
    assert len(kept) == 12 and dropped == {}


def test_top_out_veto_drops_only_a_lethal_choice_with_a_survivable_alternative():
    lethal = _records(1, run_id="a", chosen_value=-1e6, best_value=-3.5)
    forced = _records(1, run_id="b", chosen_value=-1e6, best_value=-1e6)
    fine = _records(1, run_id="c", chosen_value=-2.0, best_value=-1.0)
    kept, dropped = apply_filters(lethal + forced + fine)
    assert [r.run_id for r in kept] == ["b", "c"]
    assert dropped == {"top_out_veto": 1}


def test_fixture_run_after_filters():
    records, _ = read_run(FIXTURE)
    kept, dropped = apply_filters(records)
    # 3 records, topped out: the last 5 are dropped -> nothing survives the spiral.
    assert kept == [] and dropped == {"death_spiral": 3}


def test_split_refuses_rows_on_the_wrong_side_of_the_seed_line():
    train_ok = _records(2, seed=TRAIN_SEED_MIN)
    valid_ok = _records(2, seed=EVAL_SEEDS[0])
    stray = _records(2, seed=50)
    train, valid, excluded = split(train_ok + valid_ok + stray)
    assert {r.seed for r in train} == {TRAIN_SEED_MIN}
    assert {r.seed for r in valid} == {EVAL_SEEDS[0]}
    assert excluded == {"seed_out_of_pool": 2}
    assert all(r.seed >= TRAIN_SEED_MIN for r in train)
    assert all(r.seed in EVAL_SEEDS for r in valid)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_corpus.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'apply_filters'`.

- [ ] **Step 3: Implement**

Add to `autotune/tetris_corpus.py` after `GRADE_KEYS`:

```python
from tetris_agent.quality import TOP_OUT_VALUE  # one source of truth for the lethal sentinel

DEATH_SPIRAL = 5
TRAIN_SEED_MIN = 100
EVAL_SEEDS = (1, 2, 3, 4, 5)
```

(Move the import to the module's import block, after `from pathlib import Path`, so ruff's `I` rule is happy.)

Add after `read_run`:

```python
def apply_filters(records: list[Record], death_spiral: int = DEATH_SPIRAL) -> tuple[list[Record], Counter]:
    """Outcome-based filters, plus the one oracle veto it is safe to make.

    The grader (~490-level) is weaker than the teacher (530), so regret and rank
    are never filters. The spiral rule drops the last `death_spiral` decisions of
    a run that topped out; the veto drops a move that left the next piece nowhere
    to go when a survivable move existed.
    """
    dropped: Counter = Counter()
    by_run: dict[str, list[Record]] = {}
    for r in records:
        by_run.setdefault(r.run_id, []).append(r)

    kept: list[Record] = []
    for run_records in by_run.values():
        run_records = sorted(run_records, key=lambda r: r.turn)
        if run_records and run_records[0].outcome.get("topped_out"):
            cut = max(0, len(run_records) - death_spiral)
            dropped["death_spiral"] += len(run_records) - cut
            run_records = run_records[:cut]
        for r in run_records:
            chosen_value = r.grade.get("chosen_value", 0.0)
            best_value = r.grade.get("best_value", 0.0)
            if chosen_value <= TOP_OUT_VALUE and best_value > TOP_OUT_VALUE:
                dropped["top_out_veto"] += 1
                continue
            kept.append(r)
    return kept, dropped


def split(records: list[Record]) -> tuple[list[Record], list[Record], Counter]:
    """Train rows come only from seeds >= TRAIN_SEED_MIN; validation rows only from EVAL_SEEDS."""
    train: list[Record] = []
    valid: list[Record] = []
    excluded: Counter = Counter()
    for r in records:
        if r.seed >= TRAIN_SEED_MIN:
            train.append(r)
        elif r.seed in EVAL_SEEDS:
            valid.append(r)
        else:
            excluded["seed_out_of_pool"] += 1
    return train, valid, excluded
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_corpus.py -v && uv run ruff check autotune/tetris_corpus.py`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add autotune/tetris_corpus.py tests/test_tetris_corpus.py
git commit -F - <<'EOF'
tetris_corpus: outcome filters, the top-out veto, and the seed split

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 5: The SFT view, byte-identical to the served prompt

**Files:**
- Modify: `autotune/tetris_corpus.py`
- Test: `tests/test_tetris_corpus.py`

**Interfaces:**
- Produces: `LIVE_DEADLINE_S = 15.0`; `board_array(rows: list[str]) -> np.ndarray`; `sft_row(record: Record) -> dict` with `{"messages": [system, user, assistant]}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tetris_corpus.py`:

```python
import json

from autotune.tetris_corpus import LIVE_DEADLINE_S, board_array, sft_row


def test_sft_user_turn_is_byte_identical_to_the_served_pi_prompt():
    from tetris_agent.pi_policy import PI_JSON_INSTRUCTIONS, PI_PROMPT_SUFFIX
    from tetris_agent.prompts import build_user_prompt, legal_placements, system_prompt_for

    r, *_ = read_run(FIXTURE)[0]
    row = sft_row(r)
    system, user, assistant = row["messages"]
    board = board_array(r.board)
    expected_user = (
        build_user_prompt("features", board, r.piece, r.next_piece, legal_placements(board, r.piece), r.turn,
                          deadline_s=LIVE_DEADLINE_S)
        + PI_PROMPT_SUFFIX
    )
    assert system == {"role": "system", "content": system_prompt_for("features") + PI_JSON_INSTRUCTIONS}
    assert user == {"role": "user", "content": expected_user}
    assert assistant["role"] == "assistant"


def test_sft_assistant_turn_is_the_terse_placement_json():
    r, *_ = read_run(FIXTURE)[0]
    assistant = sft_row(r)["messages"][2]["content"]
    parsed = json.loads(assistant)
    assert parsed == {"rotation": r.chosen[0], "col": r.chosen[1], "reason": r.reason}
    assert assistant == '{"rotation": 0, "col": 3, "reason": "Flat on the floor."}'


def test_board_array_round_trips():
    r, *_ = read_run(FIXTURE)[0]
    arr = board_array(r.board)
    assert arr.shape == (18, 10) and arr.dtype == bool
    assert ["".join("#" if c else "." for c in row) for row in arr] == r.board
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_corpus.py -k "sft or board_array" -v`
Expected: FAIL with `ImportError: cannot import name 'sft_row'`.

- [ ] **Step 3: Implement**

Add to `autotune/tetris_corpus.py` imports: `import numpy as np`. Add constant `LIVE_DEADLINE_S = 15.0` beside the others. Add after `split`:

```python
def board_array(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def sft_row(record: Record) -> dict:
    """`{"messages": [system, user, assistant]}` — the format train_sft.py reads.

    Rendered through tetris's own prompt code and the pi arm's exact suffixes, so
    the training input is byte-identical to what the served student receives.
    The deadline line is the level-0 fall time even for rows minted paused: the
    deployed prompt is the live one.
    """
    from tetris_agent.pi_policy import PI_JSON_INSTRUCTIONS, PI_PROMPT_SUFFIX
    from tetris_agent.prompts import build_user_prompt, legal_placements, system_prompt_for

    harness = record.harness or "features"
    board = board_array(record.board)
    placements = legal_placements(board, record.piece)
    user = build_user_prompt(
        harness, board, record.piece, record.next_piece, placements, record.turn, deadline_s=LIVE_DEADLINE_S
    )
    assistant = json.dumps({"rotation": record.chosen[0], "col": record.chosen[1], "reason": record.reason})
    return {
        "messages": [
            {"role": "system", "content": system_prompt_for(harness) + PI_JSON_INSTRUCTIONS},
            {"role": "user", "content": user + PI_PROMPT_SUFFIX},
            {"role": "assistant", "content": assistant},
        ]
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_corpus.py -v && uv run ruff check autotune/tetris_corpus.py`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add autotune/tetris_corpus.py tests/test_tetris_corpus.py
git commit -F - <<'EOF'
tetris_corpus: SFT view rendered by tetris's own prompt code

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 6: The eval view ships the recomputed ranking

**Files:**
- Modify: `autotune/tetris_corpus.py`
- Test: `tests/test_tetris_corpus.py`

**Interfaces:**
- Produces: `eval_row(record: Record, ply: int = 2) -> dict` with keys `messages` (system + user only), `teacher` (`[rot, col]`), `ranking` (`[[rot, col, value], ...]`, best first), `turn`, `run_id`, `seed`. The tier-1 grader in the training job reads only this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tetris_corpus.py`:

```python
from autotune.tetris_corpus import eval_row


def test_eval_row_carries_prompt_teacher_and_the_full_ranking():
    from tetris_agent.prompts import legal_placements

    r, *_ = read_run(FIXTURE)[0]
    row = eval_row(r)
    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    assert row["messages"] == sft_row(r)["messages"][:2]
    assert row["teacher"] == r.chosen
    # One entry per legal placement, computed by the same enumeration the prompt shows.
    assert len(row["ranking"]) == len(legal_placements(board_array(r.board), r.piece))
    assert all(len(entry) == 3 for entry in row["ranking"])
    values = [v for _, _, v in row["ranking"]]
    assert values == sorted(values, reverse=True)
    assert tuple(r.chosen) in {(rot, col) for rot, col, _ in row["ranking"]}
    assert (row["turn"], row["run_id"], row["seed"]) == (r.turn, r.run_id, r.seed)


def test_eval_row_ranking_is_the_oracle_ranking():
    from tetris_agent.policy import Genome
    from tetris_agent.quality import rank_placements

    r, *_ = read_run(FIXTURE)[0]
    row = eval_row(r, ply=2)
    expected = rank_placements(board_array(r.board), r.piece, r.next_piece, Genome(), 2)
    assert [(rot, col) for rot, col, _ in row["ranking"]] == [p for p, _ in expected]
```

(The fixture's grade *values* are hand-written and are not asserted against the oracle anywhere; the fixture's `genome` is the default `Genome()`, which is why the second test can compare against it.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_corpus.py -k eval_row -v`
Expected: FAIL with `ImportError: cannot import name 'eval_row'`.

- [ ] **Step 3: Implement**

Add after `sft_row`:

```python
GENOME_KEYS = ("w_lines", "w_agg_height", "w_holes", "w_bumpiness")


def eval_row(record: Record, ply: int = 2) -> dict:
    """A held-out row for tier-1: the prompt, the teacher's choice, and the oracle's
    full ranking recomputed from the board — so the training job grades answers
    with no tetris dependency of its own."""
    from tetris_agent.policy import Genome
    from tetris_agent.quality import rank_placements

    genome = Genome(**{k: record.grade["genome"][k] for k in GENOME_KEYS})
    ranked = rank_placements(board_array(record.board), record.piece, record.next_piece, genome, ply)
    messages = sft_row(record)["messages"][:2]
    return {
        "run_id": record.run_id,
        "seed": record.seed,
        "turn": record.turn,
        "messages": messages,
        "teacher": list(record.chosen),
        "ranking": [[rot, col, round(value, 6)] for (rot, col), value in ranked],
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_corpus.py -v && uv run ruff check autotune/tetris_corpus.py`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add autotune/tetris_corpus.py tests/test_tetris_corpus.py tests/fixtures/tetris_run
git commit -F - <<'EOF'
tetris_corpus: eval view carries the recomputed ranking

The training job's tier-1 grader reads ranking + teacher off each held-out row,
so the job needs PyPI and the Hub and nothing else.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 7: The corpus CLI — records, views, stats, corpus id

**Files:**
- Modify: `autotune/tetris_corpus.py`
- Test: `tests/test_tetris_corpus.py`

**Interfaces:**
- Produces: `build_corpus(runs_dir: Path, out_root: Path, ply: int = 2) -> Path` writing `<out_root>/<corpus_id>/{records,train,valid,eval}.jsonl` and `stats.json`; `corpus_id(records) -> str` as `YYYYMMDD-<12 hex>`; `main(argv) -> int` for `python -m autotune.tetris_corpus --runs ../tetris/runs --out data/tetris`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tetris_corpus.py`:

```python
import shutil

from autotune.tetris_corpus import build_corpus, corpus_id


def _runs_dir(tmp_path):
    """Two copies of the fixture: one on a train seed that survives, one on eval seed 1."""
    runs = tmp_path / "runs"
    for name, seed, topped in (("train-run", 100, False), ("eval-run", 1, False)):
        dst = runs / name
        shutil.copytree(FIXTURE, dst)
        summary = json.loads((dst / "summary.json").read_text())
        summary["run_id"] = name
        summary["fitness"]["topped_out"] = topped
        (dst / "summary.json").write_text(json.dumps(summary))
        lines = (dst / "events.jsonl").read_text().splitlines()
        session = json.loads(lines[0])
        session["data"]["seed"] = seed
        lines[0] = json.dumps(session)
        (dst / "events.jsonl").write_text("\n".join(lines) + "\n")
    return runs


def test_build_corpus_writes_every_file_and_the_stats(tmp_path):
    out = build_corpus(_runs_dir(tmp_path), tmp_path / "data", ply=2)
    assert out.parent == tmp_path / "data"
    names = {p.name for p in out.iterdir()}
    assert names == {"records.jsonl", "train.jsonl", "valid.jsonl", "eval.jsonl", "stats.json"}
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(x) for x in (out / "valid.jsonl").read_text().splitlines()]
    ev = [json.loads(x) for x in (out / "eval.jsonl").read_text().splitlines()]
    # 3 records per run; the veto drops turn 3 in each -> 2 train, 2 valid.
    assert len(train) == 2 and len(valid) == 2 and len(ev) == 2
    assert all(len(row["messages"]) == 3 for row in train)
    assert all("ranking" in row for row in ev)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["runs"] == 2
    assert stats["records"] == 6
    assert stats["kept"] == 4
    assert stats["excluded"] == {"late": 2, "top_out_veto": 2}
    assert stats["per_seed"] == {"1": 2, "100": 2}
    assert stats["train_rows"] == 2 and stats["valid_rows"] == 2
    assert 0.0 <= stats["teacher"]["mean_regret_norm"] <= 1.0
    assert stats["teacher"]["top1_rate"] == 0.5


def test_corpus_id_is_dated_and_content_addressed(tmp_path):
    records, _ = read_run(FIXTURE)
    a = corpus_id(records)
    b = corpus_id(list(reversed(records)))
    assert a == b and len(a.split("-")[1]) == 12 and a[:8].isdigit()


def test_build_corpus_skips_a_run_with_no_boards(tmp_path):
    runs = _runs_dir(tmp_path)
    old = runs / "old-run"
    shutil.copytree(FIXTURE, old)
    lines = []
    for line in (old / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        e["data"].pop("board", None)
        lines.append(json.dumps(e))
    (old / "events.jsonl").write_text("\n".join(lines) + "\n")
    out = build_corpus(runs, tmp_path / "data")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["runs"] == 3 and stats["excluded"]["no_board"] == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_corpus.py -k "build_corpus or corpus_id" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add imports `import argparse`, `import hashlib`, `import sys`, `from datetime import datetime, timezone` to the module. Add after `eval_row`:

```python
def corpus_id(records: list[Record]) -> str:
    """`YYYYMMDD-<12 hex>`: the date plus a hash of the record set, order-independent."""
    keys = sorted(f"{r.run_id}:{r.turn}:{r.chosen}" for r in records)
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
    return f"{datetime.now(timezone.utc):%Y%m%d}-{digest}"


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def build_corpus(runs_dir: Path, out_root: Path, ply: int = 2) -> Path:
    """Read every run under runs_dir into <out_root>/<corpus_id>/ and return that directory."""
    runs_dir, out_root = Path(runs_dir), Path(out_root)
    all_records: list[Record] = []
    excluded: Counter = Counter()
    n_runs = 0
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file()):
        n_runs += 1
        records, ex = read_run(run_dir)
        all_records.extend(records)
        excluded.update(ex)

    kept, dropped = apply_filters(all_records)
    excluded.update(dropped)
    train, valid, stray = split(kept)
    excluded.update(stray)

    out = out_root / corpus_id(kept)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "records.jsonl", (r.to_dict() for r in kept))
    n_train = _write_jsonl(out / "train.jsonl", (sft_row(r) for r in train))
    n_valid = _write_jsonl(out / "valid.jsonl", (sft_row(r) for r in valid))
    _write_jsonl(out / "eval.jsonl", (eval_row(r, ply) for r in valid))

    per_seed: Counter = Counter(str(r.seed) for r in kept)
    regrets = [r.grade.get("regret_norm", 0.0) for r in kept]
    top1 = [1.0 if r.grade.get("rank") == 1 else 0.0 for r in kept]
    stats = {
        "corpus_id": out.name,
        "runs": n_runs,
        "records": len(all_records),
        "kept": len(kept),
        "excluded": dict(excluded),
        "per_seed": dict(sorted(per_seed.items())),
        "train_rows": n_train,
        "valid_rows": n_valid,
        "ply": ply,
        "teacher": {
            "mean_regret_norm": round(sum(regrets) / len(regrets), 6) if regrets else None,
            "top1_rate": round(sum(top1) / len(top1), 6) if top1 else None,
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tetris_corpus", description="runs/ -> placement corpus")
    ap.add_argument("--runs", default="../tetris/runs")
    ap.add_argument("--out", default="data/tetris")
    ap.add_argument("--ply", type=int, default=2)
    ap.add_argument("--hf-repo", default="bdougie/tetris-placements", help="dataset repo the upload hint names")
    args = ap.parse_args(argv)
    out = build_corpus(Path(args.runs), Path(args.out), ply=args.ply)
    stats = json.loads((out / "stats.json").read_text())
    print(json.dumps(stats, indent=2))
    print(f"\nwritten: {out}")
    print(f"upload:  hf upload {args.hf_repo} {out} {out.name} --type dataset --private")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_corpus.py -v && uv run ruff check autotune/tetris_corpus.py`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Run it against the real runs on this box, as a check that it reads them**

Run: `uv run python -m autotune.tetris_corpus --runs ../tetris/runs --out /tmp/tetris-corpus-check`
Expected: prints stats with `"runs": 16` or so, `kept: 0` (no run on disk has a board yet — Task 1 has not been exercised in a benchmark), and `excluded.no_board` equal to the number of graded decisions on disk. This proves the reader and the skip path against real files.

- [ ] **Step 6: Commit**

```bash
git add autotune/tetris_corpus.py tests/test_tetris_corpus.py
git commit -F - <<'EOF'
tetris_corpus: build_corpus writes records, views and stats under a content id

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 8: `tetris_rollout` — the proxy, the preflight, and the mint

**Files:**
- Create: `autotune/tetris_rollout.py`
- Test: `tests/test_tetris_rollout.py`

**Interfaces:**
- Produces: `bench_command(seed: int, *, model: str, max_pieces: int, harness: str = "features", effort: str = "off") -> list[str]`; `proxy_command(project: str, listen: str, upstream: str) -> list[str]`; `proxy_answers(base_url: str, timeout_s: float = 2.0) -> bool`; `mint(seeds, *, tetris_dir: Path, model: str, max_pieces: int, project: str, proxy_listen: str, upstream: str, run=subprocess.run, popen=subprocess.Popen, answers=proxy_answers) -> list[dict]` returning manifest rows `{run_id, seed, arm, project}` and writing `<tetris_dir>/runs/mint-<project>.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tetris_rollout.py`:

```python
import json
from pathlib import Path

import pytest

from autotune.tetris_rollout import bench_command, mint, proxy_answers, proxy_command


def test_bench_command_is_the_spec_invocation():
    cmd = bench_command(101, model="pi/gemma4:26b", max_pieces=100)
    assert cmd[:3] == ["uv", "run", "tetris-bench"]
    assert "--paused" in cmd and "--fixed-effort" in cmd and "--no-control" in cmd and "--no-power" in cmd
    assert cmd[cmd.index("--models") + 1] == "pi/gemma4:26b"
    assert cmd[cmd.index("--harnesses") + 1] == "features"
    assert cmd[cmd.index("--efforts") + 1] == "off"
    assert cmd[cmd.index("--seeds") + 1] == "101"
    assert cmd[cmd.index("--max-pieces") + 1] == "100"


def test_proxy_command_names_the_project():
    cmd = proxy_command("tetris-mint-x", "127.0.0.1:8092", "http://127.0.0.1:11434")
    assert cmd[:3] == ["tapes", "serve", "proxy"]
    assert cmd[cmd.index("--project") + 1] == "tetris-mint-x"
    assert cmd[cmd.index("--provider") + 1] == "openai"


def test_proxy_answers_is_false_when_nothing_listens():
    assert proxy_answers("http://127.0.0.1:1", timeout_s=0.2) is False


def test_mint_refuses_to_start_a_game_when_the_proxy_does_not_answer(tmp_path):
    started = []

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_run(cmd, **kw):
        started.append(cmd)
        raise AssertionError("tetris-bench must not run")

    with pytest.raises(RuntimeError, match="proxy"):
        mint(
            [100], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
            proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434", startup_s=0.0,
            run=fake_run, popen=lambda *a, **k: FakeProc(), answers=lambda url, timeout_s=2.0: False,
        )
    assert started == []


def test_mint_records_the_new_run_dirs_with_their_seed(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "older").mkdir()
    seen_env = {}

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_run(cmd, cwd=None, env=None, check=True, **kw):
        seen_env.update(env or {})
        seed = cmd[cmd.index("--seeds") + 1]
        d = runs / f"20260904-00000{seed}-abc"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"label": "pi/gemma4:26b/features/off+fixed"}))
        (d / "summary.json").write_text(json.dumps({"run_id": d.name, "fitness": {}}))

    rows = mint(
        [100, 101], tetris_dir=tmp_path, model="pi/gemma4:26b", max_pieces=10, project="p",
        proxy_listen="127.0.0.1:8092", upstream="http://127.0.0.1:11434",
        run=fake_run, popen=lambda *a, **k: FakeProc(), answers=lambda url, timeout_s=2.0: True,
    )
    assert [(r["seed"], r["arm"], r["project"]) for r in rows] == [
        (100, "pi/gemma4:26b/features/off+fixed", "p"),
        (101, "pi/gemma4:26b/features/off+fixed", "p"),
    ]
    assert seen_env["TETRIS_TAPES_OLLAMA_URL"] == "http://127.0.0.1:8092/v1"
    manifest = [json.loads(x) for x in (runs / "mint-p.jsonl").read_text().splitlines()]
    assert manifest == rows
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_rollout.py -v`
Expected: `ModuleNotFoundError: No module named 'autotune.tetris_rollout'`.

- [ ] **Step 3: Implement**

`autotune/tetris_rollout.py`:

```python
"""Try: mint teacher games in tetris under a tapes capture proxy.

Owns the proxy for one invocation (project `tetris-mint-<timestamp>`), refuses to
start a game unless the proxy answers, runs `tetris-bench --paused` one seed at a
time, and writes a manifest of the run directories it produced. Subprocess
wrapper; exercised by the smoke run, not unit tests, except for the pure parts.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "pi/gemma4:26b"
DEFAULT_MAX_PIECES = 100
DEFAULT_PROXY_LISTEN = "127.0.0.1:8092"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
PROXY_STARTUP_S = 10.0


def bench_command(
    seed: int, *, model: str, max_pieces: int, harness: str = "features", effort: str = "off"
) -> list[str]:
    return [
        "uv", "run", "tetris-bench",
        "--paused", "--fixed-effort", "--no-control", "--no-power",
        "--models", model,
        "--harnesses", harness,
        "--efforts", effort,
        "--seeds", str(seed),
        "--max-pieces", str(max_pieces),
    ]


def proxy_command(project: str, listen: str, upstream: str) -> list[str]:
    return [
        "tapes", "serve", "proxy",
        "--provider", "openai",
        "--upstream", upstream,
        "--listen", listen,
        "--project", project,
    ]


def proxy_answers(base_url: str, timeout_s: float = 2.0) -> bool:
    """True when the proxy forwards a GET /v1/models to Ollama and gets a 200 back."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _run_dirs(runs_dir: Path) -> set[str]:
    return {p.name for p in runs_dir.iterdir() if p.is_dir()} if runs_dir.is_dir() else set()


def mint(
    seeds,
    *,
    tetris_dir: Path,
    model: str = DEFAULT_MODEL,
    max_pieces: int = DEFAULT_MAX_PIECES,
    project: str | None = None,
    proxy_listen: str = DEFAULT_PROXY_LISTEN,
    upstream: str = DEFAULT_UPSTREAM,
    startup_s: float = PROXY_STARTUP_S,
    run=subprocess.run,
    popen=subprocess.Popen,
    answers=proxy_answers,
) -> list[dict]:
    tetris_dir = Path(tetris_dir)
    runs_dir = tetris_dir / "runs"
    project = project or f"tetris-mint-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    base_url = f"http://{proxy_listen}"

    proc = popen(proxy_command(project, proxy_listen, upstream), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + startup_s
        while not answers(base_url) and time.monotonic() < deadline:
            time.sleep(0.25)
        if not answers(base_url):
            raise RuntimeError(
                f"tapes proxy at {base_url} does not answer — is Postgres up (`tapes local up`) and "
                f"Ollama at {upstream}? Refusing to play uncaptured."
            )

        env = {**os.environ, "TETRIS_TAPES_OLLAMA_URL": f"{base_url}/v1"}
        rows: list[dict] = []
        for seed in seeds:
            before = _run_dirs(runs_dir)
            run(bench_command(seed, model=model, max_pieces=max_pieces), cwd=str(tetris_dir), env=env, check=True)
            for name in sorted(_run_dirs(runs_dir) - before):
                meta_path = runs_dir / name / "meta.json"
                label = json.loads(meta_path.read_text()).get("label", "") if meta_path.is_file() else ""
                rows.append({"run_id": name, "seed": int(seed), "arm": label, "project": project})
        with open(runs_dir / f"mint-{project}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return rows
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - subprocess driver
    ap = argparse.ArgumentParser(prog="tetris_rollout", description="mint teacher games under tapes capture")
    ap.add_argument("--tetris-dir", default="../tetris")
    ap.add_argument("--seeds", nargs="+", type=int, required=True, help="train-pool seeds, >= 100")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    low = [s for s in args.seeds if s < 100]
    if low:
        ap.error(f"seeds below 100 are the evaluation pool and are never minted: {low}")
    rows = mint(args.seeds, tetris_dir=Path(args.tetris_dir), model=args.model, max_pieces=args.max_pieces,
                project=args.project)
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_rollout.py -v && uv run ruff check autotune/tetris_rollout.py`
Expected: 5 PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add autotune/tetris_rollout.py tests/test_tetris_rollout.py
git commit -F - <<'EOF'
tetris_rollout: mint teacher games under a tapes proxy it owns

Refuses to start a game unless the proxy answers; one seed per invocation;
writes a manifest of the run dirs it produced.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

---

## Phase C — training, packaging, the gate

### Task 9: The training job — a self-contained uv script with a tested grader

**Files:**
- Create: `autotune/tetris_train_job.py`
- Test: `tests/test_tetris_train_job.py`

**Interfaces:**
- Produces: `parse_placement(text: str) -> tuple[int, int] | None`; `grade_answers(rows: list[dict], answers: list[str]) -> dict` with keys `n, top1, top3, mean_regret_norm, parse_failures, teacher_agreement`; and the job entry `main()` (GPU; smoke-tested) reading env `CORPUS_REPO`, `CORPUS_ID`, `BASE_MODEL`, `ADAPTER_REPO`, `MAX_STEPS`, `HF_TOKEN`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tetris_train_job.py`:

```python
from autotune.tetris_train_job import grade_answers, parse_placement


def test_parse_placement_reads_the_first_json_object():
    assert parse_placement('{"rotation": 2, "col": 5, "reason": "x"}') == (2, 5)
    assert parse_placement('Sure!\n{"rotation": 0, "col": 3}\ntrailing') == (0, 3)
    assert parse_placement("no json here") is None
    assert parse_placement('{"rotation": "two", "col": 5}') is None


def _rows():
    ranking = [[0, 3, -2.0], [0, 4, -2.5], [1, 0, -3.0], [1, 7, -6.0]]
    return [
        {"teacher": [0, 3], "ranking": ranking},
        {"teacher": [0, 4], "ranking": ranking},
        {"teacher": [0, 3], "ranking": ranking},
        {"teacher": [1, 7], "ranking": ranking},
    ]


def test_grade_answers_scores_rank_regret_and_agreement():
    answers = [
        '{"rotation": 0, "col": 3}',   # rank 1, regret 0, agrees with teacher
        '{"rotation": 1, "col": 0}',   # rank 3, regret (2.0-3.0... ) = 1.0/4.0 = 0.25
        "garbage",                     # parse failure
        '{"rotation": 1, "col": 7}',   # rank 4, regret_norm 1.0, agrees with teacher
    ]
    out = grade_answers(_rows(), answers)
    assert out["n"] == 4
    assert out["parse_failures"] == 1
    assert out["top1"] == 0.25
    assert out["top3"] == 0.5
    assert abs(out["mean_regret_norm"] - (0.0 + 0.25 + 1.0) / 3) < 1e-9
    assert out["teacher_agreement"] == 0.5


def test_grade_answers_treats_an_illegal_placement_as_a_parse_failure():
    out = grade_answers(_rows()[:1], ['{"rotation": 3, "col": 9}'])
    assert out["parse_failures"] == 1 and out["top1"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_train_job.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the script**

`autotune/tetris_train_job.py`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "trl",
#   "datasets",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
"""LoRA SFT of the Tetris student on HF Jobs, with tier-1 eval, in one file.

Runs as `hf jobs uv run autotune/tetris_train_job.py` — the file is uploaded on
its own, so it imports nothing from this repo. Its pure parts (`parse_placement`,
`grade_answers`) are unit-tested locally; `main` is the GPU path, smoke-tested.

Env: CORPUS_REPO (dataset), CORPUS_ID, BASE_MODEL, ADAPTER_REPO (model),
MAX_STEPS (optional), HF_TOKEN (secret).

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_placement(text: str) -> tuple[int, int] | None:
    """(rotation, col) from the first {...} in text, or None."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rot, col = d.get("rotation"), d.get("col")
    if isinstance(rot, bool) or isinstance(col, bool):
        return None
    if not isinstance(rot, int) or not isinstance(col, int):
        return None
    return rot, col


def grade_answers(rows: list[dict], answers: list[str]) -> dict:
    """Tier-1 metrics against each row's shipped ranking (best first).

    An unparseable or illegal answer is a parse failure and is excluded from the
    rank and regret means, so those numbers describe the answers that were placements.
    """
    n = len(rows)
    parse_failures = 0
    top1 = top3 = agree = 0
    regrets: list[float] = []
    for row, text in zip(rows, answers):
        choice = parse_placement(text)
        ranking = row["ranking"]
        index = next((i for i, (r, c, _) in enumerate(ranking) if (r, c) == choice), None) if choice else None
        if index is None:
            parse_failures += 1
            continue
        rank = index + 1
        top1 += rank == 1
        top3 += rank <= 3
        agree += list(choice) == list(row["teacher"])
        best, worst, value = ranking[0][2], ranking[-1][2], ranking[index][2]
        span = best - worst
        regrets.append((best - value) / span if span > 0 else 0.0)
    graded = n - parse_failures
    return {
        "n": n,
        "parse_failures": parse_failures,
        "top1": top1 / n if n else 0.0,
        "top3": top3 / n if n else 0.0,
        "teacher_agreement": agree / n if n else 0.0,
        "mean_regret_norm": sum(regrets) / graded if graded else None,
    }


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        sys.exit(f"[train_job] missing env {name}")
    return value


def main() -> int:  # pragma: no cover - GPU path, exercised by the smoke job
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    corpus_repo = _env("CORPUS_REPO", "bdougie/tetris-placements")
    corpus_id = _env("CORPUS_ID")
    base_model = _env("BASE_MODEL", "google/gemma-4-E4B-it")
    adapter_repo = _env("ADAPTER_REPO", "bdougie/gemma-4-E4B-tetris-lora")
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))

    corpus = Path(snapshot_download(corpus_repo, repo_type="dataset", allow_patterns=[f"{corpus_id}/*"])) / corpus_id
    train_rows = [json.loads(x) for x in (corpus / "train.jsonl").read_text().splitlines() if x.strip()]
    eval_rows = [json.loads(x) for x in (corpus / "eval.jsonl").read_text().splitlines() if x.strip()]
    print(f"[train_job] corpus {corpus_id}: {len(train_rows)} train rows, {len(eval_rows)} eval rows")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Fail here, not after an hour, if the template cannot take a system turn.
    print("[train_job] rendered example:\n" + tokenizer.apply_chat_template(train_rows[0]["messages"], tokenize=False)[:600])

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    projections = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    try:
        # A multimodal checkpoint keeps its text stack under `language_model`; target only that.
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                 task_type="CAUSAL_LM",
                                                 target_modules=rf".*language_model.*\.({projections})$"))
    except ValueError:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=projections.split("|")))
    model.print_trainable_parameters()

    out_dir = Path("adapter")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir), num_train_epochs=2, max_steps=max_steps,
            per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4,
            warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True, max_length=4096,
            gradient_checkpointing=True, logging_steps=10, report_to="none", save_strategy="no",
        ),
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Tier 1: generate a placement per held-out board and grade it against the shipped ranking.
    model.eval()
    answers: list[str] = []
    for row in eval_rows:
        ids = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=64, do_sample=False)
        answers.append(tokenizer.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True))
    tier1 = grade_answers(eval_rows, answers)
    tier1["tokens_per_answer"] = (
        sum(len(tokenizer(a)["input_ids"]) for a in answers) / len(answers) if answers else None
    )
    tier1["corpus_id"] = corpus_id
    (out_dir / "eval_tier1.json").write_text(json.dumps(tier1, indent=2))
    print("[train_job] tier-1: " + json.dumps(tier1))

    api = HfApi()
    api.create_repo(adapter_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=adapter_repo, path_in_repo="adapter",
                      commit_message=f"adapter from corpus {corpus_id}")
    api.create_tag(adapter_repo, tag=corpus_id, tag_message=f"corpus {corpus_id}", exist_ok=True)
    print(f"[train_job] pushed {adapter_repo} tag {corpus_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run to verify the pure tests pass**

Run: `uv run pytest tests/test_tetris_train_job.py -v && uv run ruff check autotune/tetris_train_job.py`
Expected: 3 PASS; ruff clean. (Importing the module locally must not import torch — the heavy imports live inside `main`.)

- [ ] **Step 5: Accept the Gemma license once**

Open `https://huggingface.co/google/gemma-4-E4B-it` while logged in as the account behind `hf auth whoami` and accept the terms. The job's `HF_TOKEN` is that account's token; without acceptance the job fails at `from_pretrained` with a 401.

- [ ] **Step 6: Commit**

```bash
git add autotune/tetris_train_job.py tests/test_tetris_train_job.py
git commit -F - <<'EOF'
tetris_train_job: LoRA SFT + tier-1 eval as one self-contained uv script

Depends on PyPI and the Hub only, so it runs under `hf jobs uv run` with no git
checkout. parse_placement and grade_answers are unit-tested; main is the GPU
path the smoke job exercises.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 10: The packaging job — merge, GGUF, Q4_K_M, upload

**Files:**
- Create: `autotune/tetris_package_job.py`

**Interfaces:**
- Produces: a uv script for `hf jobs uv run --flavor cpu-xl` reading env `BASE_MODEL`, `ADAPTER_REPO`, `CORPUS_ID`, `HF_TOKEN`; uploads `gemma-4-E4B-tetris-<corpus_id>-Q4_K_M.gguf` to `ADAPTER_REPO` at path `gguf/`. No unit tests: it is a subprocess/CPU driver (in the coverage omit list from Task 2); the smoke job in Task 12 is its test.

- [ ] **Step 1: Write the script**

`autotune/tetris_package_job.py`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "huggingface_hub",
#   "gguf>=0.10",
#   "sentencepiece",
#   "protobuf",
#   "safetensors",
# ]
# ///
"""Package the tuned student for Ollama: merge the LoRA, convert to GGUF, quantize.

Runs as `hf jobs uv run --flavor cpu-xl autotune/tetris_package_job.py`. The
quant is Q4_K_M on purpose — the same quant the baseline `gemma4:latest` is
served at, so the tier-2 comparison measures training, not quantization.

Env: BASE_MODEL, ADAPTER_REPO, CORPUS_ID, HF_TOKEN (secret).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def sh(*cmd: str, cwd: str | None = None) -> None:
    print("[package_job] $ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=cwd, check=True)


def main() -> int:  # pragma: no cover - CPU/subprocess driver, exercised by the smoke job
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-E4B-it")
    adapter_repo = os.environ.get("ADAPTER_REPO", "bdougie/gemma-4-E4B-tetris-lora")
    corpus_id = os.environ["CORPUS_ID"]

    adapter = Path(snapshot_download(adapter_repo, revision=corpus_id, allow_patterns=["adapter/*"])) / "adapter"
    print(f"[package_job] merging {adapter_repo}@{corpus_id} into {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained("merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained("merged")

    sh("apt-get", "update")
    sh("apt-get", "install", "-y", "--no-install-recommends", "git", "cmake", "build-essential")
    sh("git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp")
    sh("cmake", "-B", "build", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF", cwd="llama.cpp")
    sh("cmake", "--build", "build", "--target", "llama-quantize", "-j", cwd="llama.cpp")

    f16 = "merged-f16.gguf"
    quant = f"gemma-4-E4B-tetris-{corpus_id}-Q4_K_M.gguf"
    sh(sys.executable, "llama.cpp/convert_hf_to_gguf.py", "merged", "--outfile", f16, "--outtype", "f16")
    sh("llama.cpp/build/bin/llama-quantize", f16, quant, "Q4_K_M")

    api = HfApi()
    api.upload_file(path_or_fileobj=quant, path_in_repo=f"gguf/{quant}", repo_id=adapter_repo,
                    commit_message=f"Q4_K_M GGUF for corpus {corpus_id}")
    print(f"[package_job] uploaded {adapter_repo}/gguf/{quant}")
    print(f"[package_job] on the box: ollama pull hf.co/{adapter_repo}:Q4_K_M && "
          f"ollama cp hf.co/{adapter_repo}:Q4_K_M gemma4-e4b-tetris:{corpus_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check autotune/tetris_package_job.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add autotune/tetris_package_job.py
git commit -F - <<'EOF'
tetris_package_job: merge, convert, quantize to Q4_K_M, upload

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

### Task 11: The tier-2 gate, and registering a tuned tag

**Files:**
- Create: `autotune/tetris_eval.py`, `scripts/register_pi_tag.py`
- Test: `tests/test_tetris_eval.py`
- Modify (tetris): `src/tetris_agent/pricing.py` — one `MODELS` line per tuned tag, done at gate time (Step 6)

**Interfaces:**
- Produces: `GateResult` dataclass (`passed: bool, reasons: list[str], tuned_median, base_median, tuned_tokens, base_tokens, tuned_late, base_late`); `gate(tuned_runs: list[dict], base_runs: list[dict]) -> GateResult` where each run dict is one entry of a `tetris-bench` result file's `runs` list; `runs_for(result: dict, model: str) -> list[dict]` selecting runs whose `arm` starts with `f"{model}/"`; `main(argv)` for `python -m autotune.tetris_eval --results <json> --tuned pi/<tag> --base pi/gemma4:latest`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tetris_eval.py`:

```python
from autotune.tetris_eval import gate, runs_for


def _run(arm, seed, score, pieces, tokens, late):
    return {
        "arm": arm, "seed": seed, "error": None, "policy_stats": {},
        "fitness": {"score": score, "pieces_placed": pieces,
                    "policy": {"tokens_per_decision": tokens, "late": late}},
    }


BASE = "pi/gemma4:latest/features/off+live+fixed"
TUNED = "pi/gemma4-e4b-tetris:x/features/off+live+fixed"


def _base():
    return [_run(BASE, s, 180, 30, 34.0, 0) for s in (1, 2, 3, 4, 5)]  # race 330 each


def test_gate_passes_when_all_three_rules_hold():
    tuned = [_run(TUNED, s, 380, 30, 40.0, 0) for s in (1, 2, 3, 4, 5)]  # race 530
    r = gate(tuned, _base())
    assert r.passed and r.reasons == []
    assert (r.tuned_median, r.base_median) == (530.0, 330.0)


def test_gate_fails_on_an_equal_median():
    tuned = [_run(TUNED, s, 180, 30, 34.0, 0) for s in (1, 2, 3, 4, 5)]
    r = gate(tuned, _base())
    assert not r.passed and any("median" in x for x in r.reasons)


def test_gate_uses_the_median_not_one_lucky_seed():
    tuned = [_run(TUNED, 1, 900, 30, 34.0, 0)] + [_run(TUNED, s, 100, 30, 34.0, 0) for s in (2, 3, 4, 5)]
    assert not gate(tuned, _base()).passed


def test_gate_token_rule_is_inclusive_at_one_point_five():
    at = [_run(TUNED, s, 380, 30, 51.0, 0) for s in (1, 2, 3, 4, 5)]   # 34 * 1.5 = 51.0
    over = [_run(TUNED, s, 380, 30, 51.1, 0) for s in (1, 2, 3, 4, 5)]
    assert gate(at, _base()).passed
    r = gate(over, _base())
    assert not r.passed and any("tokens" in x for x in r.reasons)


def test_gate_fails_when_late_decisions_increase():
    tuned = [_run(TUNED, s, 380, 30, 34.0, 1) for s in (1, 2, 3, 4, 5)]
    r = gate(tuned, _base())
    assert not r.passed and any("late" in x for x in r.reasons)


def test_runs_for_selects_by_model_prefix():
    result = {"runs": _base() + [_run(TUNED, 1, 1, 1, 1.0, 0)]}
    assert len(runs_for(result, "pi/gemma4:latest")) == 5
    assert len(runs_for(result, "pi/gemma4-e4b-tetris:x")) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tetris_eval.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tetris_eval.py`**

```python
"""Tier-2 gate: the live benchmark over held-out seeds, tuned against its own base.

Reads a `tetris-bench` result file (data/benchmarks/benchmark-*.json in the
tetris repo). Three rules, all required: median race beats the baseline's
median; tokens per decision <= 1.5x the baseline's; late decisions do not
increase. The goal (>= 530) is reported, not gated.

Spec: docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RATIO = 1.5
GOAL_RACE = 530.0


def race(run: dict) -> float:
    """tetris_agent.fitness.race_score, restated so this module needs no tetris import at eval time."""
    f = run.get("fitness") or {}
    return float(f.get("score", 0)) + 5.0 * float(f.get("pieces_placed", 0))


def _tokens(run: dict) -> float:
    return float(((run.get("fitness") or {}).get("policy") or {}).get("tokens_per_decision", 0.0))


def _late(run: dict) -> int:
    return int(((run.get("fitness") or {}).get("policy") or {}).get("late", 0))


def runs_for(result: dict, model: str) -> list[dict]:
    prefix = f"{model}/"
    return [r for r in result.get("runs", []) if str(r.get("arm", "")).startswith(prefix) and not r.get("error")]


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    tuned_median: float = 0.0
    base_median: float = 0.0
    tuned_tokens: float = 0.0
    base_tokens: float = 0.0
    tuned_late: int = 0
    base_late: int = 0

    def to_dict(self) -> dict:
        return {**self.__dict__, "goal_race": GOAL_RACE, "goal_met": self.tuned_median >= GOAL_RACE}


def gate(tuned_runs: list[dict], base_runs: list[dict]) -> GateResult:
    if not tuned_runs or not base_runs:
        return GateResult(False, ["no runs for tuned or base arm"])
    r = GateResult(
        passed=True,
        tuned_median=statistics.median(race(x) for x in tuned_runs),
        base_median=statistics.median(race(x) for x in base_runs),
        tuned_tokens=statistics.fmean(_tokens(x) for x in tuned_runs),
        base_tokens=statistics.fmean(_tokens(x) for x in base_runs),
        tuned_late=sum(_late(x) for x in tuned_runs),
        base_late=sum(_late(x) for x in base_runs),
    )
    if not r.tuned_median > r.base_median:
        r.reasons.append(f"median race {r.tuned_median:.0f} does not beat baseline {r.base_median:.0f}")
    if r.tuned_tokens > TOKEN_RATIO * r.base_tokens + 1e-9:
        r.reasons.append(f"tokens/decision {r.tuned_tokens:.1f} > {TOKEN_RATIO} x baseline {r.base_tokens:.1f}")
    if r.tuned_late > r.base_late:
        r.reasons.append(f"late decisions rose {r.base_late} -> {r.tuned_late}")
    r.passed = not r.reasons
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tetris_eval", description="tier-2 gate over a tetris-bench result file")
    ap.add_argument("--results", required=True, help="data/benchmarks/benchmark-*.json from the tetris repo")
    ap.add_argument("--tuned", required=True, help="pi/<tuned tag>")
    ap.add_argument("--base", default="pi/gemma4:latest")
    args = ap.parse_args(argv)
    result = json.loads(Path(args.results).read_text())
    verdict = gate(runs_for(result, args.tuned), runs_for(result, args.base))
    print(json.dumps(verdict.to_dict(), indent=2))
    print("PASS" if verdict.passed else "FAIL: " + "; ".join(verdict.reasons))
    return 0 if verdict.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_tetris_eval.py -v && uv run ruff check autotune/tetris_eval.py`
Expected: 6 PASS; ruff clean.

- [ ] **Step 5: Write the tag registrar**

`scripts/register_pi_tag.py`:

```python
"""Add an Ollama tag to ~/.pi/agent/models.json the way every local entry is listed.

    uv run python scripts/register_pi_tag.py gemma4-e4b-tetris:20260905-ab12cd

Backs the file up to the next free models.json.bakN first. Idempotent.
"""

import json
import os
import shutil
import sys


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    tag = argv[1]
    path = os.path.expanduser("~/.pi/agent/models.json")
    n = 1
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    shutil.copy2(path, f"{path}.bak{n}")
    with open(path) as f:
        data = json.load(f)
    models = data["providers"]["ollama"]["models"]
    if any(m.get("id") == tag for m in models):
        print(f"already registered: {tag}")
        return 0
    models.append({"id": tag, "contextWindow": 131072, "input": ["text"], "reasoning": True,
                   "thinkingLevelMap": {"off": "none"}})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"registered {tag} (backup: {path}.bak{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 6: Document the tetris-side registration (executed at gate time, per tuned tag)**

In `../tetris/src/tetris_agent/pricing.py`, beside `"pi/gemma4:latest"`, add one line per tuned tag:

```python
    "pi/gemma4-e4b-tetris:<corpus_id>": ModelSpec("pi/gemma4-e4b-tetris:<corpus_id>", 0.0, 0.0, True, 0),
```

with the real corpus id substituted. An unlisted tag is deliberately effort-free in tetris and would run with thinking on. Commit it in tetris on the `feat/graded-board` branch with the message `pricing: list the tuned tetris tag <corpus_id>`.

- [ ] **Step 7: Commit**

```bash
git add autotune/tetris_eval.py tests/test_tetris_eval.py scripts/register_pi_tag.py
git commit -F - <<'EOF'
tetris_eval: the three-rule tier-2 gate, and a pi tag registrar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

---

## Phase D — prove every hop, then run it

### Task 12: Smoke, not unit — every hop once, small

**Files:** none new. This task is a checklist that proves the pipeline end to end before the five-hour mint. Every command is exact; every expected output is stated.

- [ ] **Step 1: Bring up the capture stack**

```bash
nc -z localhost 5432 || tapes local up            # Postgres
curl -sf http://localhost:11434/api/tags >/dev/null || echo "start ollama"
```

Expected: both reachable.

- [ ] **Step 2: Mint 2 tiny runs on train seeds**

```bash
uv run python -m autotune.tetris_rollout --tetris-dir ../tetris --seeds 100 101 --max-pieces 20
```

Expected: two JSON manifest rows printed; two new dirs under `../tetris/runs/`; `runs/mint-tetris-mint-*.jsonl` written. Then verify capture happened:

```bash
PGPASSWORD=tapes psql -h 127.0.0.1 -U tapes -d tapes -c "select count(*) from raw_turns where received_at > now() - interval '15 minutes';"
```

Expected: about 40 (one per decision). If 0, the proxy env did not reach pi — stop and fix before minting anything larger.

- [ ] **Step 3: Mint 1 tiny run on an eval seed**

`tetris_rollout` refuses seeds below 100 on purpose (`seeds below 100 are the evaluation pool`), so eval-seed runs are always minted directly with tetris:

```bash
cd ../tetris && uv run tetris-bench --paused --fixed-effort --no-control --no-power \
  --models pi/gemma4:26b --harnesses features --efforts off --seeds 1 --max-pieces 20 && cd -
```

Expected: one new run dir with seed 1.

- [ ] **Step 4: Build and upload the corpus**

```bash
uv run python -m autotune.tetris_corpus --runs ../tetris/runs --out data/tetris
```

Expected: stats with `kept` around 45–55, `per_seed` containing `"1"`, `"100"`, `"101"`, `train_rows` ≈ 35, `valid_rows` ≈ 18, `excluded.no_board` equal to the old runs' graded count. Copy the `upload:` line it prints and run it, e.g.:

```bash
hf upload bdougie/tetris-placements data/tetris/<corpus_id> <corpus_id> --type dataset --private
```

Expected: upload succeeds; `hf datasets list --author bdougie` shows the repo.

- [ ] **Step 5: A 20-step training job on `l4x1`**

```bash
hf jobs uv run --flavor l4x1 --timeout 1h \
  --secrets HF_TOKEN=$(hf auth token) \
  --env CORPUS_ID=<corpus_id> --env MAX_STEPS=20 \
  autotune/tetris_train_job.py
```

Expected: the rendered example prints (system, user with the board, assistant JSON); `print_trainable_parameters` shows a LoRA of a few tens of millions of params; 20 steps; a `tier-1:` JSON line with `n` equal to `valid_rows`, `parse_failures` reported; `pushed bdougie/gemma-4-E4B-tetris-lora tag <corpus_id>`. If `apply_chat_template` raises on the system role, the fix is in `sft_row`: merge the system content into the user turn — but confirm first against Ollama's `TEMPLATE` for `gemma4:latest` (`ollama show --modelfile gemma4:latest`) so training matches what is served.

- [ ] **Step 6: Package on `cpu-xl`**

```bash
hf jobs uv run --flavor cpu-xl --timeout 2h \
  --secrets HF_TOKEN=$(hf auth token) \
  --env CORPUS_ID=<corpus_id> \
  autotune/tetris_package_job.py
```

Expected: merge, clone, cmake build, convert, quantize, upload; final line `on the box: ollama pull ...`.

- [ ] **Step 7: Serve it here and register it**

```bash
ollama pull hf.co/bdougie/gemma-4-E4B-tetris-lora:Q4_K_M
ollama cp hf.co/bdougie/gemma-4-E4B-tetris-lora:Q4_K_M gemma4-e4b-tetris:<corpus_id>
uv run python scripts/register_pi_tag.py gemma4-e4b-tetris:<corpus_id>
```

Then the tetris pricing line from Task 11 Step 6, committed. Verify effort lands:

```bash
cd ../tetris && uv run tetris-bench --paused --fixed-effort --no-control --no-power \
  --models pi/gemma4-e4b-tetris:<corpus_id> --harnesses features --efforts off --seeds 1 --max-pieces 5
```

Expected: arm label ends in `/off+fixed` and `tok_s`/latency imply ~35 tokens per decision. A label without `/off` means the pricing line is missing.

- [ ] **Step 8: One-seed tier-2 dry run**

```bash
cd ../tetris && uv run tetris-bench --fixed-effort --no-control \
  --models pi/gemma4:latest pi/gemma4-e4b-tetris:<corpus_id> \
  --harnesses features --efforts off --seeds 1 --max-pieces 30 && cd -
uv run python -m autotune.tetris_eval --results ../tetris/data/benchmarks/benchmark-<ts>.json \
  --tuned pi/gemma4-e4b-tetris:<corpus_id>
```

Expected: a verdict JSON and PASS/FAIL. With a 20-step adapter the verdict is expected to be FAIL; the point is that every hop ran and the gate read the file.

- [ ] **Step 9: Record the smoke in the plan's own commit**

```bash
git add -A && git commit -F - <<'EOF'
Smoke: every hop of the tetris distillation pipeline ran once

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

(If the smoke produced no tracked changes, skip the commit.)

### Task 13: Corpus v1, the real run, and the write-ups

- [ ] **Step 1: Mint corpus v1**

```bash
uv run python -m autotune.tetris_rollout --tetris-dir ../tetris --seeds $(seq -s ' ' 100 159) --max-pieces 100
```

Expected: 60 runs, ≈ 5 h. Then mint the five eval seeds at 30 pieces directly with `tetris-bench` (Step 3 of Task 12, once per seed 1–5).

- [ ] **Step 2: Build, upload, train (full), package, register, gate**

Run Task 12 steps 4–8 with `MAX_STEPS` unset and `--flavor l40sx1` for training, and seeds `1 2 3 4 5` in the tier-2 benchmark. Record the gate's JSON.

- [ ] **Step 3: Amend the spec's two sentences the plan changed**

In `docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md`:

Replace, in *The gate — two tiers*, the clause "with `tetris_agent` installed into the job from the tetris repo as a dependency" with "against the ranking each eval row ships (recomputed by `tetris_corpus.eval_row`), so the job depends on PyPI and the Hub only".

Replace, in *Training*, "`train_sft.py`'s existing `cuda` path — TRL/PEFT bf16 LoRA — driven by a uv script that runs under `hf jobs uv run`" with "a self-contained uv script (`autotune/tetris_train_job.py`) that mirrors `train_sft.py`'s CUDA path — TRL/PEFT bf16 LoRA — and runs under `hf jobs uv run` with no git checkout".

Add under the *Revised 2026-09-04* block a second block:

```markdown
> **Revised 2026-09-04, during planning.** HF Jobs scripts depend only on PyPI and the Hub.
> Tier-1 grades against a ranking shipped in each eval row rather than importing
> `tetris_agent` into the job, and training is a self-contained script mirroring
> `train_sft.py`'s CUDA path rather than importing it. Same observable outputs; no
> private-repo access from a cloud job.
```

- [ ] **Step 4: README section**

Append to `README.md`:

```markdown
## Tetris: distilling a local student (2026-09)

The same Try → Check → Reward → Nudge shape, pointed at [`pcc-labs/tetris`](../tetris):
`autotune/tetris_rollout.py` mints `gemma4:26b` games under a tapes proxy, `tetris_corpus.py`
turns `runs/` into a corpus whose prompts are rendered by tetris's own `build_user_prompt`,
`tetris_train_job.py` LoRA-tunes `google/gemma-4-E4B-it` on HF Jobs (~$2 on `l40sx1`),
`tetris_package_job.py` merges and quantizes it back to a Q4_K_M GGUF for Ollama, and
`tetris_eval.py` gates it on the live benchmark over seeds 1–5. Design:
[docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md](docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md).

    uv sync --extra tetris
    uv run python -m autotune.tetris_rollout --seeds 100 101 --max-pieces 100
    uv run python -m autotune.tetris_corpus --runs ../tetris/runs --out data/tetris
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN=$(hf auth token) --env CORPUS_ID=<id> autotune/tetris_train_job.py
```

- [ ] **Step 5: Benchmark write-up in tetris**

Create `../tetris/benchmarks/<date>-e4b-distilled.md` in the style of `benchmarks/2026-09-03-local-reasoning-matrix.md`: the tier-2 table (both arms, five seeds, race / score / lines / pieces / late / avg holes / tok per decision / s per decision / regret / top1 / top3), the gate verdict, the tier-1 numbers, the corpus id and stats, and one paragraph on whether the score moved because placements improved.

- [ ] **Step 6: Commit both repos**

```bash
git add README.md docs/superpowers/specs/2026-09-04-tetris-placement-distillation-design.md
git commit -F - <<'EOF'
Tetris distillation: README section and spec amendments from planning

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
cd ../tetris && git add benchmarks/ src/tetris_agent/pricing.py && git commit -F - <<'EOF'
bench: the distilled E4B on the live screen, seeds 1-5

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EgFSNTqg31oPZJfVXjuQZL
EOF
```

---

## Self-review notes

- **Spec coverage.** Contract (one field) → Task 1. Record and reading rules → Task 3. Filters, veto, seed split → Task 4. SFT view byte-identical incl. pi suffixes → Task 5. Eval view → Task 6. Corpus layout, `corpus_id`, stats, upload → Task 7. Rollout owning the proxy, preflight, project, manifest → Task 8. Training on HF Jobs, `l40sx1`, `l4x1` fallback, tier-1 inside the job → Tasks 9 and 12. Packaging at Q4_K_M → Task 10. Tier-2 three rules, goal reported not gated, registration in both `models.json` and `pricing.MODELS` → Task 11. Smoke before the mint → Task 12. Corpus v1 and write-ups → Task 13. Non-goals are respected: no DPO, no ExIt loop, no benchmark or prompt change.
- **Two deliberate deviations from the spec's wording**, both recorded in Task 13 Step 3: tier-1 grades against a shipped ranking instead of importing tetris into the job, and training is a self-contained script instead of importing `train_sft.train`. Reason: an HF Job must not depend on a git checkout of either repo.
- **Type consistency.** `Record` fields are used by name in Tasks 4–7; `eval_row` keys (`teacher`, `ranking`) match `grade_answers` in Task 9; `gate` reads `fitness.policy.tokens_per_decision` and `.late`, which is the shape `tetris-bench` writes (checked against `data/benchmarks/benchmark-20260904-135533.json`).
- **Fixture values are hand-written.** The fixture's grade numbers are chosen to exercise the filters (a veto case, a top-out, a late decision), not to match the oracle, and no test compares them against `rank_placements`. Task 6's tests compare the recomputed ranking against the oracle directly, which is the property that matters.
