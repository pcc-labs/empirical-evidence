# Tetris placement distillation: a local adapter as good as cloud, on less power

**Status:** approved in discussion 2026-09-04, not yet implemented.
**Companion repo:** [`pcc-labs/tetris`](../../../../tetris) — the game, the benchmark, the grader.

## Goal

Fine-tune `google/gemma-4-E4B-it` — served as `gemma4:latest`, 9.6 GB, 330 race on the live
screen — to play Game Boy Tetris as well as `gemma4:26b` — 19 GB, 530 race, the cloud median —
at ~35 tokens per decision with no reasoning. Cloud-level play at half the resident memory and
faster decisions: *as good as cloud, on less power and cost.*

Record every decision in a form that a later Expert Iteration loop can train a learned
evaluator from, so this corpus is the first round of that loop rather than a dead end.

## Why this framing

Three facts from the 2026-09-03 matrices and the 2026-09-04 screening runs decided it.

**Local parity with the cloud median already happened.** On live gravity, seed 1, 30-piece
cap, `features` harness, effort pinned:

| arm | served | race | tok/decision | s/decision | late | avg holes |
|---|---|---|---|---|---|---|
| `gemma4:26b` / off (local) | 19 GB | **530** | 33 | 3.04 | 1 | 0.00 |
| `deepseek-v4-pro`, `gemma4-31b`, `gpt-oss:20b low` (cloud) | — | 530 | | | | |
| `gemma4:latest` E4B / off (local) | 9.6 GB | **330** | 34 | 2.18 | 0 | 10.20 |
| heuristic control | — | 490 | | | | |
| `nemotron-3-super` / off (cloud) | — | 730 | 28 | 2.38 | 0 | 0.20 |

The gap is not to "cloud". A 19 GB local MoE sits on the cloud median with the cleanest board
in either table. The remaining win is the same score from a smaller model.

**On live gravity, local models cannot buy quality with reasoning.** Every local arm at `low`
or `medium` flatlines at 55: `gemma4:latest / low` spends 997 tokens and 72 s per decision
against a ~15 s fall. The accidental thinking-on E4B run on 09-04 makes the point exactly:

| E4B, paused, seed 1 | top-1 | top-3 | regret | holes | s/decision |
|---|---|---|---|---|---|
| thinking on (default) | 0.43 | 0.77 | 0.231 | 3.8 | **18.5** |
| effort `off` | 0.12 | 0.48 | 0.212 | 13.5 | 1.9 |

The quality is in the weights when the model reasons; it just cannot afford to reason on the
live screen. The only route to quality on live is to bake it in so the answer comes out in
~35 tokens. That is what supervised fine-tuning on `(board → placement)` does, and it is why
"low or no reasoning" and "plays well live" are one goal, not two.

**The lookahead oracle is a grader, not a teacher.** Graded against the two-ply oracle
(`tetris/src/tetris_agent/quality.py`), 30 pieces, seed 1:

| policy | n | regret | top-1 | top-3 |
|---|---|---|---|---|
| `lookahead` (the oracle itself) | 30 | 0.000 | 1.000 | 1.000 |
| `gemma4:26b` / off | 29 | 0.035 | 0.759 | 0.897 |
| heuristic control | 30 | 0.090 | 0.533 | 0.933 |
| `gemma4:latest` E4B / off | 25 | 0.212 | 0.120 | 0.480 |
| `smollm3:3b` Q8, thinking off | 18 | 0.310 | 0.056 | 0.222 |
| random | 28 | 0.333 | 0.036 | 0.107 |

The one-ply heuristic scores 490 on live; the local `gemma4:26b` scores 530 and agrees with the
oracle more often than the heuristic does. Distilling the oracle into that model would be
regression. The oracle's job here is the one it is good at: grading every decision on the
same scale, and vetoing the blunders it is certain about.

**SmolLM3-3B is ruled out as the student.** It is what this repo already trains and packages,
so it was screened first. With quantization raised to Q8 and thinking genuinely off it answers
in format, survives 18 pieces, and places pieces indistinguishably from random. Format
compliance was fixable; spatial signal was never there. The student has to be a model that
already demonstrably plays — and E4B does.

## Non-goals

- Preference or RL training (DPO, GRPO). SFT on a 530-level teacher is the first move; the
  neutral record below is enough to add either later without replaying a game.
- Building the Expert Iteration loop. This design prepares for it (state, policy target, value
  target, an injectable evaluator) and stops there.
- Beating `nemotron-3-super` at 730. The target is 530 from a 9.6 GB model.
- Any change to how the tetris benchmark runs or what it scores. Existing rows stay comparable.
- Prompt or harness changes in tetris.
- A message broker. JSONL on disk is the seam, as both repos already state.

## Architecture: two repos, one seam

`empirical-evidence` already relates to `pokemon-kafka` this way — it drives the agent from
outside, reads its telemetry, and never modifies it. Tetris gets the same treatment.

```
tetris (this box)     play games → runs/<id>/events.jsonl (graded events carry the board) + summary.json
                      tapes capture per run (traces for builders, not the corpus)
        │ tetris_rollout.py
        ▼
empirical-evidence    tetris_corpus.py: runs/ → records.jsonl → {train,valid}.jsonl
        │ hf upload
        ▼
HF dataset            bdougie/tetris-placements (private)
        │ hf jobs uv run --flavor l40sx1        (~$2)
        ▼
HF model              bdougie/gemma-4-E4B-tetris-lora  (adapter + merged GGUF)
        │ ollama pull hf.co/...:Q4_K_M
        ▼
tetris (this box)     tetris-bench, live, held-out seeds — the gate
```

**Why the split lands where it does.** Tetris owns the game and the grader; both are pure and
deterministic, so anything derivable from a board is recomputed in `empirical-evidence` rather
than recorded. Everything still being figured out — corpus shape, filters, training recipe,
gate — lives in `empirical-evidence`, against frozen tetris output. A wrong corpus is a
converter fix and a re-run over existing run directories, not a replay of games.

**Why HF Jobs.** This box has no training path: no ROCm (`/opt/rocm` absent), no torch, and the
RTX 5090 is disconnected. Ollama reaches the Strix Halo iGPU through Vulkan, an inference
path PyTorch cannot use. HF Jobs is enabled on the account (smoke job `tetris-jobs-smoke`
returned `JOB OK` on `cpu-basic` after credits were added) and `hf jobs uv run` runs a uv
script on a GPU flavor, which matches this repo's uv-native workflow directly. So: **this box
generates data and runs evals; HF Jobs trains.**

**Two stores, not interchangeable.** `runs/` is the dataset of record: board, decision, grade,
outcome, for every arm including the non-model ones. Tapes is the high-fidelity trace layer:
the exact bytes on the wire, the full reasoning text, token and latency breakdown — what
answers "did `effort=low` buy anything" and what you read when a training round goes wrong.
Tapes has no board, no grade, no outcome, and cannot see `heuristic`, `random`, `lookahead`
or human arms. They join on `run_id` + `turn`. Neither substitutes for the other.

## The tetris contract

One field. The `placement_graded` event gains the board the decision was made on:

```python
# events.py
def build_graded_event(turn, grade, board=None):
    data = grade.to_dict()
    if board is not None:
        data["board"] = ["".join("#" if c else "." for c in row) for row in board]
    return _envelope("placement_graded", turn, data)
```

and the two call sites that already hold the board pass it:
`collector.graded(g, board=board)` in `agent.py` and `live_agent.py`. About six lines and one
test. No new file, flag, module or seam.

- `board`: 18 rows of 10 chars, `.` empty / `#` settled, row 0 at top — the encoding
  `situation_corpus.py` already uses, so its `_captured()` reader parses it unchanged.
- Safe to add: `placement_graded` shipped in #17 the day before this design, and nothing reads
  its contents — tests assert its presence and ordering, `traces.py` skips it, and the
  benchmark asserts it never reaches the viewer.
- Late, timed-out and fallback decisions are never graded, so they never carry a board. That
  is the filter the corpus wanted anyway, and it means the record below needs no join.
- `--no-quality` already suppresses the event, so it suppresses the board with it.

> **Revised 2026-09-04, after review.** The first draft asked tetris for three things: a
> sibling `boards.jsonl` with its own writer and a join on `turn`; a `--require-capture`
> flag; and a `value_fn` keyword on `rank_placements` for a future learned evaluator. The
> reasons given did not hold. The sibling file protected an "isolation guarantee" on an event
> that is a day old and has no consumers. The capture preflight belongs to whoever brings up
> the proxy, which is `tetris_rollout.py` here. And a default keyword argument costs the same
> three lines whenever it is added, so adding it before a learned evaluator exists is
> speculation. Tetris keeps its shape; everything else moved into this repo.

Not in the contract, by design: `deadline_s`. Corpus v1 mints paused, where it is `null`, and
the SFT view renders a fixed live deadline (below). If live-minted rows are ever wanted, the
deadline is a second optional field on the same event, added then.

What tetris looks like after: every module unchanged in shape, one event with one more field,
and a line in `pricing.MODELS` per tuned tag — the existing way every arm is added.

## The record

`tetris_corpus.py` reads `events.jsonl` (`piece_spawn`, `placement_decision`,
`placement_graded`) and `summary.json` into one neutral row per **graded** decision:

```
run_id, turn, arm, model, harness, effort, seed, mode
board, piece, next_piece                           ← state, as shown to the model
chosen: [rotation, col], reason                    ← what the arm did
grade: {rank, legal_count, regret, regret_norm, best, chosen_value, best_value,
        worst_value, genome, ply}                  ← verbatim from placement_graded
outcome: {final_score, lines, pieces_placed, topped_out, pieces_after}
```

`pieces_after = pieces_placed − turn`: how long the game survived from this state. It is a
subtraction, it is free, and it is the value target Expert Iteration needs. It is the one
field that cannot be reconstructed later.

**Nothing else is stored.** `rank_placements(board, piece, next_piece, genome, ply)` is pure,
so the full ranked distribution is recomputable offline at any depth — the policy target for
Expert Iteration, the rejected set for a preference view, a deeper grade if two-ply proves
too shallow. Recording it would freeze one depth into the corpus for no gain.

Reading rules:
- One row per `placement_graded` event; `board`, `chosen` and the grade come from that event,
  `piece` and `next_piece` from the `piece_spawn` with the same `turn`, `reason` from the
  `placement_decision` with the same `turn`. A decision that was never graded (late,
  timed-out, fallback) has no event and yields no row; the exclusion counts are reported in
  `stats.json`.
- `reason` is the teacher's one-sentence explanation and is the assistant turn's text in the
  SFT view.
- A `placement_graded` event without a `board` field (everything recorded before this
  change) is skipped with a count, never replayed: `traces.mine_run` recovers a board on 33 %
  of graded decisions, and that path is not good enough to train on.

## Teacher, student, and minting the corpus

**Teacher: `gemma4:26b` at effort `off`, `features` harness.** Local, free, 530 on live,
top-1 0.759 against the oracle, 0.00 holes. Same family as the student — same tokenizer,
same chat template, same prompt conventions — which is the easiest distillation there is and
sidesteps the class of failure that ruled out SmolLM3. `nemotron-3-super` (730, ≈$1 per
30,000 decisions through `pi/`) is the documented stronger teacher if E4B saturates against
the 26B; it is not in v1.

**Student: `google/gemma-4-E4B-it`.** 8.0 B parameters (E4B is effective, not total), served
at Q4_K_M as `gemma4:latest`. Baseline on live: 330 race, 34 tok/decision, 2.18 s, 0 late.
Baseline graded: top-1 0.12, top-3 0.48, regret 0.212. It already plays, already answers
inside the budget; the whole gap is quality.

**Minting runs paused, not live.** Live is gravity-bound at ~15 s per piece regardless of how
fast the teacher answers, so 30,000 decisions on live is ~125 hours. Paused is bound by the
teacher's 3 s, and because the teacher answers in 3 s with 0–1 late decisions on live, its
paused decisions are the same decisions it would make live — only the deadline line in the
prompt differs, and the SFT view restores that (below). Corpus v1:

- `tetris_rollout.py` shells out to `tetris-bench --paused --models pi/gemma4:26b
  --harnesses features --efforts off --fixed-effort --max-pieces 100`, one seed per
  invocation, seeds from the **train pool** (`100, 101, …`). It owns the tapes stack for the
  invocation: brings up the proxy on its own port (`:8093`, not `tapes-up.sh`'s `:8092`) with
  `--project tetris-mint-<timestamp>`, refuses to start if that port is already bound,
  **checks that the proxy answers before starting any game**, and after the first seed
  **verifies capture with a direct Postgres count** (`SELECT count(*) FROM raw_turns WHERE
  provider = 'openai' AND received_at > <mint_start>`) rather than the tapes read API, which
  this invocation never starts. Sets `TETRIS_TAPES_OLLAMA_URL` and tears the proxy down after,
  with its stderr saved to `<runs_dir>/mint-<project>.proxy.log`. Arms within an invocation are
  serial (one model resident on the iGPU), so a project + time window identifies a run and the
  `Piece {turn}.` header identifies the decision.
- 60 runs × 100 pieces ≈ 6,000 decisions before filtering, ≈ 5 h of iGPU time. Longer games
  than the benchmark's 30-piece cap on purpose: mid- and late-game boards are the ones a
  330-level student fails on.
- `max_pieces 100` rather than "to game over": the teacher rarely tops out, and a bounded run
  is a bounded mint.

**Filter on outcome, not on the oracle.** Rejection-sampling only works when the grader is
stronger than what it grades, and this grader (~490-level) is weaker than the teacher (530).
So:

- Drop the final 5 decisions of any run that topped out (the death spiral). Keep everything
  else from runs that survived to `max_pieces`.
- The one oracle veto: drop a decision whose `chosen_value == TOP_OUT_VALUE` while
  `best_value > TOP_OUT_VALUE` — a move that left the next piece nowhere to go when a
  survivable move existed. The oracle is certain about those regardless of its overall
  strength.
- `regret` and `rank` are kept as columns and reported in `stats.json`. They are not filters.

**Seed split, stated so it cannot drift.** Seed 1 is the historical benchmark seed and is
**never trained on**. Held-out evaluation seeds are `{1, 2, 3, 4, 5}`. Training seeds start at
100. `tetris_corpus.py` refuses to emit a training row from a seed below 100 and refuses to
emit a validation row from a seed at or above 100.

## The training views

Thin adapters over `records.jsonl`; v1 emits only the first.

**SFT** — `{"messages": [system, user, assistant]}`, the format `train_sft.py` already reads:

- `system`: `prompts.system_prompt_for("features")`, verbatim from tetris.
- `user`: `prompts.build_user_prompt("features", board, piece, next_piece, placements, turn,
  deadline_s=15.0)`, called from the tetris package, never reimplemented. Training input is
  byte-identical to inference input, and a prompt change regenerates the corpus instead of
  silently rotting it. `deadline_s=15.0` is the level-0 fall time — the live prompt shape,
  even though the rows were minted paused. Every row renders the same fixed value; the event
  carries no per-decision deadline (see the contract).
- `assistant`: `{"rotation": r, "col": c, "reason": "<teacher's sentence>"}` — the terse
  target. Tokens per decision is a gated number (below); a corpus that taught long answers
  would fail that gate.

**Preference** (not in v1): `{prompt, chosen, rejected}` with `rejected` drawn from the
recomputed ranking below a `regret_norm` margin.

**Expert Iteration** (not in v1): `{board, piece, next_piece, policy_target, value_target}`
with the policy target a softmax over recomputed ranked values and the value target
`pieces_after`.

Corpus layout: `data/tetris/<corpus_id>/{records,train,valid}.jsonl` + `stats.json`
(row counts, exclusions by reason, per-seed counts, regret/top-1 distribution of the
teacher). `corpus_id` is `YYYYMMDD-<short hash of the record set>`, and the whole directory
is uploaded to the HF dataset repo under that id, so a job names exactly the corpus it ran on.

## Training

`train_sft.py`'s existing `cuda` path — TRL/PEFT bf16 LoRA — driven by a uv script that runs
under `hf jobs uv run`:

- Base: `google/gemma-4-E4B-it`. Flavor: `l40sx1` (48 GB, $1.80/h) — an 8 B bf16 base with a
  LoRA fits without QLoRA; expected well under an hour. `l4x1` with 4-bit QLoRA is the
  cheaper fallback, not the default, so the first result is not confounded by quantized
  training.
- Config: the existing profile defaults for rank, alpha, dropout, LR schedule; target modules
  are all linear projections. Nothing tuned in v1 — the point of v1 is a clean baseline.
- Thinking: E4B at `off` already answers in ~35 tokens, so the served template is left as it
  is. `_patch_chat_template_disable_thinking` is SmolLM3-specific and is not applied.
- The job downloads `bdougie/tetris-placements/<corpus_id>`, trains, and pushes the adapter to
  `bdougie/gemma-4-E4B-tetris-lora` tagged with the corpus id. Secrets via `--secrets HF_TOKEN`.

## Packaging back to Ollama

A second job (`cpu-xl` suffices): merge the adapter into the base, convert with llama.cpp's
`convert_hf_to_gguf.py`, quantize to **Q4_K_M** — the same quant the baseline `gemma4:latest`
is served at, so the comparison measures training and not quantization — and push the GGUF to
the same model repo, one GGUF per corpus. On this box: `scripts/deploy_gguf.sh <corpus_id>` —
not `ollama pull hf.co/<repo>:Q4_K_M`, which selects by quant tag alone and, from the second
corpus onward, resolves *some* Q4_K_M file in the repo rather than this one. The script
downloads the named file, `ollama create`s it as `gemma4-e4b-tetris:<corpus_id>`, and lists it
in `~/.pi/agent/models.json` (with `thinkingLevelMap.off = "none"`, as every local entry); it
still has to be added by hand to `tetris/pricing.MODELS` (`supports_effort=True`). An unlisted
tag is deliberately effort-free in tetris and would run with thinking on — the 09-04 E4B
mis-run was exactly that.

## The gate — two tiers

**Tier 1, every checkpoint, no emulator.** Runs inside the training job on the adapter
directly — the pattern `eval_heldout.py` already uses — so no checkpoint has to be packaged to
GGUF just to be measured. On the held-out records (seeds 1–5) the student generates a
placement for each board; the answer is graded with `quality.grade` against the recomputed
ranking, with `tetris_agent` installed into the job from the tetris repo as a dependency.
Reported: top-1, top-3, mean `regret_norm`, parse failures, tokens per answer, and agreement
with the teacher's chosen placement. Minutes on the same GPU, pure Python around the grader.

**Tier 2, final, the benchmark itself.** `tetris-bench` on live gravity, level 0, `features`,
effort `off`, `--fixed-effort`, seeds `{1, 2, 3, 4, 5}`, 30-piece cap — the exact screen every
published row is on. Both the tuned tag and the untouched `gemma4:latest` baseline, same
invocation.

Pass, all three required:

1. **Median race over the five seeds beats the baseline's median** on the same seeds. Never
   one seed: the placement-quality design records 225 versus 530 on identical weights and
   seed.
2. **Tokens per decision ≤ 1.5 × baseline** (baseline ≈ 34). A student that learned to talk
   more would be slower on live, and the whole premise is speed.
3. **Late decisions do not increase.**

Goal, distinct from the gate: median race ≥ 530 — the cloud median, from a 9.6 GB model. A run
that passes the gate but lands short of the goal is progress and gets the next corpus; a run
that fails the gate is a corpus or recipe problem and does not ship.

The `regret` / `top1` / `top3` columns in the tier-2 table are reported alongside, so the
write-up can say whether the score moved because placements improved.

## Expert Iteration: what this design preserves

Not built here; listed so nothing in v1 forecloses it.

- **State** — `board`, `piece`, `next_piece` per decision.
- **Policy target** — the full ranked distribution, recomputable at any depth from the state.
- **Value target** — `pieces_after`, recorded per row.
- **The search** — `rank_placements` is already the search. A learned evaluator replaces
  `_value` behind a keyword argument when one exists; three lines then, not now.
- **Re-labeling** — because state is stored, any later teacher (a deeper oracle, a learned
  evaluator, a stronger model) can re-label the same boards offline.

The accurate name is Expert Iteration, not AlphaZero: Tetris is single-player with a stochastic
piece stream, so there is no self-play opponent and no zero-sum value. The mechanism — search
plus learned evaluator, iterated, with the search's output as the next round's label — is what
exceeds a hand-written teacher, and it is the same mechanism.

## Risks

**Compounding error.** The teacher's boards are clean (0.00 holes). The student will make
mistakes the teacher never made and then face boards the corpus never contained. This is the
standard imitation-learning failure and the first thing to check if tier 1 is strong and tier
2 is not. The remedy is DAgger-shaped: mint a second corpus from the *student's* trajectories
with the teacher answering on the student's boards. The neutral record supports it unchanged;
it is corpus v2, not v1.

**Student capacity.** E4B may saturate below 530. The learning curve across corpus sizes
(1k, 3k, 6k rows) says whether more data helps; if it plateaus, the levers are the stronger
teacher (`nemotron-3-super`) or the next student up (`google/gemma-4-12B-it`, ~24 GB served —
which gives back most of the footprint win and needs saying out loud).

**Quantization parity.** Tuned and baseline must be compared at the same quant. Stated above;
restated here because it is the easiest way to fool this whole exercise.

**Effort actually off.** Two of the four screening runs on 09-04 ran with thinking on by
accident — an unlisted tag, and a template that hardcoded `/think`. Every eval row must show
tokens per decision in the ~35 range or it is not the arm it claims to be.

**The oracle as veto.** `TOP_OUT_VALUE` vetoes are safe; anything broader would import the
grader's weakness. If the veto count in `stats.json` is more than a handful per thousand
rows, something upstream is wrong.

## Costs

| step | where | cost |
|---|---|---|
| mint corpus v1 (60 × 100 pieces) | this box, iGPU | ~5 h, no spend |
| train | HF Jobs `l40sx1` | ~$2 |
| package | HF Jobs `cpu-xl` | <$1 |
| tier 1 eval | this box | minutes |
| tier 2 eval (5 seeds × 2 arms, live) | this box | ~80 min wall-clock |

## Testing

Test-driven, each watched failing first.

**tetris**
1. A `placement_graded` event published through the agent carries a `board` of 18 rows of
   10 chars that round-trips through `situation_corpus._captured`, and equals the board the
   grader was called with. `build_graded_event(turn, grade)` with no board is byte-identical
   to today's event.

**empirical-evidence**
2. `tetris_corpus` reads a synthetic run into rows with the right `pieces_after`, and reports
   late, fallback and ungraded decisions as counted exclusions.
3. A `placement_graded` event without a `board` yields no row and a counted skip, never a
   replay.
4. The top-out veto drops exactly the `TOP_OUT_VALUE`-with-alternative rows.
5. The death-spiral filter drops exactly the last 5 rows of a topped-out run.
6. The seed guard refuses a train row from seed < 100 and a validation row from seed ≥ 100.
7. The SFT view's user turn equals `build_user_prompt(...)` from the tetris package for the
   same inputs — a byte comparison, not a resemblance.
8. The SFT assistant turn parses as a placement and its `rotation`/`col` equal `chosen`.
9. `tetris_rollout` refuses to start a game when the proxy does not answer, before the
   emulator is touched.
10. Tier-1 grading of the teacher's own choices reports regret 0 and top-1 1.0.
11. The tier-2 gate returns pass/fail per the three rules on synthetic result rows, including
    the equal-median and the 1.5× boundary cases.

**Smoke, not unit:** one end-to-end mint of 2 runs × 20 pieces, corpus build, a 20-step
training job on `l4x1`, packaging, and a 1-seed tier-2 run — proving every hop before the
5-hour mint is started.
