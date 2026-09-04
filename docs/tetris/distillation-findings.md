# Tetris placement distillation, run 1: the pipeline works, the corpus does not yet

First end-to-end execution of the design in
[the design spec](../superpowers/specs/2026-09-04-tetris-placement-distillation-design.md), trained
locally on the RTX 5090 rather than on HF Jobs. Every stage has now executed on
real data. The short answer on the question the project exists to ask — does
training a 9.6 GB student on a 19 GB teacher's placements make it play better
Tetris? — is **not from this corpus**, and the tier-1 numbers say why.

## The run

| | |
|---|---|
| corpus | `20260904-8eccf25ae599` — 487 kept decisions, 366 train / 121 eval rows |
| provenance | 27 run dirs scanned, 10 contributed; 8 seeds (100-105 train, 1-2 eval), 60 pieces each |
| teacher | `gemma4:26b` at effort off, one arm: `features/off+fixed` |
| student | `google/gemma-4-E4B-it`, bf16 LoRA r=16 on the text stack's linear projections |
| recipe | batch 4 x accumulation 4, 2 epochs, gradient checkpointing |
| hardware | RTX 5090, 32 GB — 8 min 37 s, no OOM, no 4-bit |
| final train loss | 0.567, mean token accuracy 0.898 |

## Tier-1: the tuned student against its own untuned base

Same 121 held-out boards, same grader, same prompts. The untuned column was
measured separately after the run; without it the tuned column has nothing to be
better than, and the design did not originally call for it.

| metric | untuned E4B | tuned | teacher (the ceiling) | gap closed |
|---|---|---|---|---|
| top-1 | 46 (0.380) | 47 (0.388) | 50 (0.413) | 1 of 4 rows |
| top-3 | 78 (0.645) | 85 (0.702) | 98 (0.810) | 7 of 20 rows |
| mean normalized regret | 0.173 | 0.156 | 0.129 | ~39% |
| teacher agreement | 72 (0.595) | 88 (0.727) | 121 by definition | 16 of 49 rows |
| tokens per answer | 30.5 | 31.8 | — | — |
| parse failures | 0 | 0 | — | — |

The teacher column is `gemma4:26b`'s own choice scored against the same
two-ply oracle on the same 121 boards — the ceiling imitation can reach, not a
published headline. It is far below the 0.759 top-1 of the benchmark's 30-piece
row: at 60 pieces the boards are genuinely hard, and the teacher gets the
oracle's best move only 50 times in 121.

**Result: it moved toward the teacher, roughly a third of the way, and the
teacher is not far enough ahead to matter.** Against the ceiling the student can
actually reach, training closed about 39% of the regret gap and 7 of the 20
available top-3 rows. That is a real effect, not noise. But top-1 moved by a
single row out of the 4 that were available, and 4 rows is the entire distance
between the untuned student and its teacher on this metric. The student is
learning to imitate; imitation is working; the thing being imitated is only
slightly better than the starting point.

Two structural reasons this was the expected outcome, both known before the run:

1. **The corpus is 6% of the designed size.** 366 rows against the ~6,000 the
   design calls for (60 seeds x 100 pieces). The handoff predicted this run
   would fail its gate for exactly this reason.
2. **Imitation caps at the teacher's ceiling, and the ceiling is low.** Over
   all 487 kept decisions the teacher scores top-1 0.458 and regret 0.149
   (recorded in the corpus's own `stats.json`), against its published
   0.759 / 0.035 on 30-piece boards. Late-game boards are harder than anything
   the benchmark headline measures. A student that perfectly imitated this
   teacher would still only reach 0.413 top-1 on the eval boards, so most of
   the distance to a *better* Tetris player is not reachable by imitation of
   this teacher at all.

## What this run does establish

Every stage of the pipeline has executed on real data, which was the actual
blocker the 5090 handoff was written to clear:

- Train on the corpus of record, locally, in under nine minutes for ~$0.
- Tier-1 grading all 121 held-out rows with zero parse failures — the student
  emits well-formed placement JSON in ~32 tokens.
- Adapter and tier-1 eval pushed to `bdougie/gemma-4-E4B-tetris-lora`, tagged
  with the corpus id.
- Merge, GGUF conversion and Q4_K_M quantization to a servable 5.3 GB model.

## Bugs this run surfaced

**Tier-1 returned `n=0` on every run before it was fixed.** After
`SFTTrainer.train()` the model's `forward` no longer carries `logits_to_keep` in
its signature, so `generate()` receives `[batch, seq, vocab]` logits and dies in
`_sample` with "Tensors must have same number of dimensions: got 2 and 3" on the
first row. Every run would have reported a student that scored zero,
indistinguishable from a genuinely bad one. The job now grades the adapter as
saved on a freshly loaded base, and `generation_error` / `eval_rows` travel with
the metrics so a crash can never again be read as a result.

## Before trusting a tier-2 (live benchmark) number

Ollama's `gemma4:latest` baseline is 8.0 B parameters at 9.6 GB and carries the
vision and audio towers; the packaged student is the text stack alone, 7.5 B at
5.3 GB. Both are nominally Q4_K_M but they are not the same build. If the tuned
arm loses on the live gate, package the *untuned* base through the same
converter and race that against `gemma4:latest` first — otherwise a packaging
difference will be misread as a training result.

## Prognosis: train harder, or train differently?

More of the same buys a little; a different objective buys more; and the data
for the different objective already exists.

**Why "train harder" on this recipe has a low ceiling.** Supervised fine-tuning
on the teacher's choices can only converge toward the teacher, and on these
boards the teacher is 4 top-1 rows ahead of the untuned student. That is the
entire prize. Longer training on 366 rows would overfit rather than help — token
accuracy is already 0.90. The bigger corpus is still worth one run: it costs
nothing but five hours and tests the imitation hypothesis cleanly. Expect it to
pull tier-1 toward 0.41 and the live score partway from 330 toward 530,
probably not all the way.

**The gap that matters is not in the tier-1 table.** Per decision, untuned E4B
looks close to the teacher. Live, it loses 330 to 530. The difference is
distribution shift: the eval boards are boards the teacher created for itself,
while the student plays on boards its own earlier mistakes created, and small
per-decision regret compounds over 60 pieces. Tier-1 on teacher boards will
always understate that. A corpus of the student's own boards, labelled by the
oracle, closes it — that is the design's round 2 and it should move up.

**A different objective is the real lever, and it is cheap.** In rising order
of cost:

1. **Preference pairs from the oracle ranking.** Every record already carries
   the full ranked list of legal placements. Pair the best against a bad one
   and train with DPO. The student learns to rank placements rather than copy
   one label, and the ceiling becomes the oracle instead of the teacher. Same
   corpus, one new view, one TRL trainer swap.
2. **On-policy data.** Roll the student out, label its boards with the oracle,
   retrain. Fixes the compounding directly.
3. **GRPO with the oracle as reward.** Sample several placements per board,
   score each with the two-ply oracle, reinforce the good ones. No teacher at
   all. Feasible on the 5090 with LoRA on an 8 B model, slower per step.

**One caution about what would be optimised.** Tier-1 measures agreement with
a two-ply heuristic that the *winning* teacher disagrees with 59% of the time.
It is a weak proxy for the live race score. Whichever objective is chosen, the
live gate on held-out seeds is the number that counts; the oracle is a reward
signal to be validated against it, not ground truth.

## Next

Mint the full corpus (60 seeds x 100 pieces, ~5 h on the iGPU, no spend) and
retrain before reading anything into the gate. The teacher is on the box
(`gemma4:26b`, pulled 2026-09-04) and `scripts/tetris_train_local.sh` runs the
whole pipeline locally, so each additional round costs time rather than dollars
— which is the point of moving training in-house.

But note what 16x more of the same data can and cannot buy. More rows should
push the student further along the line it is already on, toward the teacher.
That line ends at 0.413 top-1 on these boards. If the goal is a student that
plays *better* than what is being imitated, the corpus size is not the only
lever that has to move — the design's own roadmap (a second corpus from the
student's own trajectories, then Expert Iteration against the oracle rather
than the teacher) is where a ceiling above the teacher's would come from. Run
the bigger corpus first, because it is cheap and it tests the imitation
hypothesis cleanly, but read its result against 0.413 rather than against 530.
