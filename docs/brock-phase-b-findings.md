# Phase B — Brock loop findings (2026-07-05)

Sibling to the Pewter/forest-crossing work (Phase A). This records the first end-to-end run of the
`autotune.loop --mode brock` loop against a **real captured pre-Brock state**, plus the tooling
that was needed to capture it.

## TL;DR (verified — corrects an earlier premature "Brock beaten" claim)

- **Brock is NOT actually beaten by any config tested — and the loop's reported "win" is a false
  positive.** The top L30 leaderboard entry shows `won=true, reward 11.76, damage_frac 1.0`, but
  that rollout's **own fitness is `brock_won=None, battles_won=0`** and a direct replay shows the
  agent **stalling against Brock's Geodude** (Scratch does 7/33, then the battle freezes). The
  reward is fooled by a spurious pokemon-kafka telemetry event: `battle_end {type 2, won:true,
  map 54, opp_lvl 12}` appears even though the agent never won. `brock.py` trusts that
  `battle_end.won` over the agent's authoritative `brock_won`/`battles_won` fields → false win.
- **The damage gradient is real and useful.** Organic **L14 Charmander** clearly loses (best
  reward `1.30`, ~15% team damage, faints ~12 turns). Higher levels deal more: L22 ≈ 0.23, L26
  reached `2.00` (one of Brock's two Pokémon KO'd). So level genuinely helps — Charmander is just
  a poor matchup (Fire resisted 0.5× by Onix, Scratch vs Onix's base-160 Def).
- **Two real blockers to a clean win (both upstream of the loop):**
  1. **Reward/telemetry false-positive** — the spurious `battle_end won:true` must be reconciled
     against `brock_won`/`battles_won` before the reward bands can be trusted.
  2. **Battle stall-guard** — pokemon-kafka's wild-flee stall guard misfires in the un-fleeable
     trainer fight: after healing it picks "run" (can't flee Brock), freezing the battle at a
     constant HP with `Action: fight` and no move executing. This caps how far any genome gets.
- **Upstream implication (feeds Phase A):** even once those are fixed, the damage curve suggests a
  much higher level (or a super-effective lead, not Charmander) is needed — Phase A must grind far
  past the organic level. This matches the skill's "if Phase B can't win at the organic level, the
  fix is upstream" hypothesis.

## The capture problem (and fix)

The loop needs a pre-Brock state, but reaching Brock is blocked by two things the baked agent
can't do:

1. **Pewter City navigation.** After the parcel quest the agent has `target=None` and no
   gym-seeking behaviour — it wanders into a house (map 58) and jams. The real **gym-door warp is
   at map-2 tile (16,17)** (read from the RAM warp table; `references/routes.json` had the wrong
   (16,11)). A map-2 A* pilot + wall-follow escape reliably reaches and enters it.
2. **Gym interior.** `game_area_collision()` returns **all-walls only on a corrupt mid-warp save**
   — from a settled state it works (40/90 walkable). The path to Brock is straight up column 4
   past a Jr.-Trainer Camper (Diglett + tanky **Sandshrew**, whose Def makes Scratch do 5/33).

The reusable capture tool (`scratchpad/solve_gym4.py` in the origin session): from the clean gym
state, poke the lead level (`autotune.party.set_lead_level`) + stock 40 Potions
(`set_bag`), climb up with live-collision (never down/out), keep the lead alive through the Camper
attrition with an HP top-up (traversal aid only — the captured state is an honest full-HP
battle-start), and save `pre_brock.state` at `enemy_level>=12`. Level-variant captures land in
`states/brock/lead_lv<N>.state`, which the loop consumes as a ladder.

## Artifacts

- `states/pre_brock.state` — organic L14 battle-start capture (loses, ~15% dmg).
- `states/brock/lead_lv{18,22,26,30}.state` — level ladder (damage rises with level; none is a
  *verified* win — the L30 "win" is the reward false-positive above).
- `out/best_brock.json` — best-by-reward config (L30) — do NOT treat as a confirmed win.
- `out/brock/leaderboard_ladder.json` — the per-level ladder leaderboard.

## Next steps to an actual win

1. Fix the false-positive: in `brock.py`, prefer the agent's `brock_won`/`battles_won` and only
   fall back to `battle_end.won` when they're absent; or fix pokemon-kafka to stop emitting the
   spurious trainer `battle_end won:true`.
2. Fix the stall-guard so it never picks an un-executable "run" in a trainer battle.
3. Re-run the ladder; if still losing, push the lead well past L30 or change the lead species.
