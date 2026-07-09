"""Genome-driven landmark-follower for Viridian Forest — the capture half of the forest SFT bridge.

``scratch_forest_follow.py`` drives the forest by headings (the nav wall the genome can't fix) but
HARDCODES battle (flee every wild, fight every trainer), so different genomes produce identical
runs. This module keeps the heading-driven navigation and **delegates every battle to the agent's
genome-driven** ``choose_action`` (via ``run_battle_turn``): wild encounters flee-or-fight by
``hp_run_threshold``, healing by ``hp_heal_threshold``, moves by the scoring weights. So crossings
differ by how far the genome SURVIVES — the spread that turns the flat forest reward into a real
gradient (see ``forest_story``).

It emits a minimal telemetry stream in the ``game_events`` shape ``forest_story.score_forest``
reads (``overworld`` map_id, ``battle_outcome {battle_type, won}``, ``discovery`` sign text,
``map_change`` on exit), so the same pure scorer and the forest SFT builder consume it with no
coupling to pokemon-kafka's collector. The genome arrives via ``EVOLVE_PARAMS`` (env), exactly
like the agent's own seam.

IO/emulator wrapper: smoke-tested via a genome sweep, not unit-tested (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

FOREST = 51
EXIT_MAPS = (13, 2)  # Route 2 / Pewter — anything past the forest (mirrors forest_story)
PERP = {"up": ["left", "right"], "down": ["left", "right"],
        "left": ["up", "down"], "right": ["up", "down"]}

# Beats from the walkthrough: (heading, slide-bias toward this perpendicular first, advance-when).
# advance: "wall" = blocked after sliding; "bag" = bag count increased; "sign" = Trainer Tips read;
# "map" = left the forest. Headings are the general direction; wall-following handles the trees.
BEATS = [
    ("left", "up", "wall"),     # 1a: walk left from entrance to the wall
    ("up", "left", "wall"),     # 1b: head north
    ("left", "up", "wall"),     # 1c: turn left again
    ("up", "right", "bag"),     # 2+3: head north up the path -> pick up Poke Ball
    ("up", "left", "bag"),      # 4+5: past trainers, stick LEFT -> Antidote
    ("left", "down", "wall"),   # 6: into the grassy area, loop
    ("up", "left", "sign"),     # 7a: north until the Trainer Tips sign
    ("left", "up", "wall"),     # 8a: head west to the far-left edge
    ("up", "left", "map"),      # 8b: walk north -> exit gate to Route 2
]

_BATTLE_TURN_CAP = 60  # hard cap on run_battle_turn calls per encounter (avoid Struggle deadlock)
# Ground-truth crossing path minted by the flood-fill that first crossed (scratch_flood3.py).
ROUTE_DEFAULT = str(Path(__file__).parent / "routes" / "forest_cross_path.json")


def _dir_to(a: tuple[int, int], b: tuple[int, int]) -> str:
    """Direction of the dominant axis from a toward b (exact for adjacent tiles)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy) and dx:
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _import_pk():
    """Make pokemon-kafka's ``scripts/`` importable (cwd=pk dir, or POKEMON_KAFKA_DIR)."""
    try:
        from agent import PokemonAgent  # noqa: F401
        from world_map import WorldMap  # noqa: F401
        return PokemonAgent, WorldMap
    except ImportError:
        from dotenv import load_dotenv

        load_dotenv()
        pk = os.environ.get("POKEMON_KAFKA_DIR")
        if not pk:
            raise RuntimeError("POKEMON_KAFKA_DIR unset and pokemon-kafka scripts not importable")
        sys.path.insert(0, str((Path(pk).resolve() / "scripts")))
        from agent import PokemonAgent  # noqa: E402
        from world_map import WorldMap  # noqa: E402
        return PokemonAgent, WorldMap


def _start_recorder(ag, record_dir: str, label: str | None, frame_interval: int):
    """Wire pokemon-kafka's ``RunRecorder`` to this crossing's PyBoy screen.

    Produces a self-contained viewer-format run (``events.jsonl`` + ``frames/*.png`` +
    ``meta.json`` + ``summary.json``) so the Paper Traces viewer shows real Game Boy frames for a
    forest crossing that is otherwise headless and frameless. ``recorder`` is importable once
    ``_import_pk`` has put pokemon-kafka's ``scripts/`` on ``sys.path``.
    """
    from recorder import RunRecorder  # noqa: E402 — pk/scripts on sys.path via _import_pk()

    run_id = RunRecorder.new_run_id(datetime.now(), "forest")
    rec = RunRecorder(
        run_id,
        Path(record_dir),
        frame_grabber=lambda: ag.pyboy.screen.image,
        frame_interval=frame_interval,
    )
    rec.start({"label": label or "Viridian Forest crossing"})
    return rec


def follow_once(
    rom: str,
    in_state: str,
    genome: dict,
    max_steps: int = 1500,
    worldmap_in: str | None = None,
    worldmap_out: str | None = None,
    out_state: str | None = None,
    shot: str | None = None,
    route: str | None = None,
    record_dir: str | None = None,
    label: str | None = None,
    frame_interval: int = 10,
) -> dict:
    """Run one genome-driven forest crossing. Returns ``{events, fitness, crossed, ...}``.

    Navigation is hand-driven; battles are handed to the agent's genome-driven policy, so
    ``genome`` (which becomes ``EVOLVE_PARAMS`` for the agent) is what varies survival between
    runs. ``events`` is the game_events-shaped stream ``forest_story.score_forest`` reads.

    With ``route`` (a JSON file with a ``path`` list of [x, y] tiles — see
    ``states/forest_cross_path.json``, extracted from the ground-truth flood-fill that crossed),
    navigation follows that tile path to the (1,0) exit warp and on through the north gate to
    Route 2. Without it, the legacy walkthrough ``BEATS`` + exit dash drive (which is known to
    wedge — the dash oscillates on the far-left column).
    """
    os.environ["EVOLVE_PARAMS"] = json.dumps(genome)  # ingested by PokemonAgent at construction
    PokemonAgent, WorldMap = _import_pk()

    ag = PokemonAgent(rom, strategy="low")
    with open(in_state, "rb") as f:
        ag.pyboy.load_state(f)
    if worldmap_in and Path(worldmap_in).exists():
        try:
            ag.world = WorldMap.load(worldmap_in)
        except Exception:  # noqa: BLE001 — a bad/old worldmap shouldn't kill the run
            pass

    events: list[dict] = []
    step = {"n": 0}
    rec = _start_recorder(ag, record_dir, label, frame_interval) if record_dir else None

    def emit(evt: dict):
        """Append a forest event; when recording, also feed the viewer run."""
        events.append(evt)
        if rec is not None:
            rec.on_event(evt)

    def ow():
        s = ag.memory.read_overworld_state()
        return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)

    last_map = {"id": None}

    def note_map(map_id):
        if map_id != last_map["id"]:
            emit({"event_type": "overworld", "turn": step["n"],
                  "data": {"map_id": map_id}})
            if last_map["id"] is not None:
                emit({"event_type": "map_change", "turn": step["n"],
                      "data": {"prev_map": last_map["id"], "new_map": map_id}})
            last_map["id"] = map_id

    def bt():
        return getattr(ag.memory.read_battle_state(), "battle_type", 0)

    hp_heal = float(genome.get("hp_heal_threshold", 0.25))  # heal below this HP fraction
    trainers = {"won": 0}
    heal_dbg = {"n": 0}
    _dbgdir = os.environ.get("AUTOTUNE_HEAL_DEBUG")

    def _dshot(tag: str):
        if _dbgdir and heal_dbg["n"] < 2:
            ag.pyboy.tick(3, True)
            img = ag.pyboy.screen.image
            if img is not None:
                img.resize((img.width * 3, img.height * 3)).save(
                    f"{_dbgdir}/heal{heal_dbg['n']}_{tag}.png"
                )

    def _heal_turn(bag_index: int):
        """Drive in-battle potion use ourselves — pk's own item action is a no-op (selects the item
        but never applies it; HP stays frozen and the bag is never decremented). Gen-1 flow:
        ITEM corner -> pick the potion -> 'use on which POKéMON?' -> the lead (first) -> clear text.
        """
        _dshot("0_inbattle")
        # Back out of any open submenu (the turn often sits in the move list) to the top-level
        # FIGHT/PKMN/ITEM/RUN menu — otherwise the cursor presses below land in the move list and
        # pick an attack instead of ITEM.
        ag.controller.press("b")
        ag.controller.wait(30)
        ag.controller.press("b")
        ag.controller.wait(30)
        # Normalize to FIGHT (top-left), then step to ITEM (bottom-left) and open the bag. Generous
        # waits — the battle-menu cursor drops short presses.
        ag.controller.press("up")
        ag.controller.wait(20)
        ag.controller.press("left")
        ag.controller.wait(20)
        ag.controller.press("down")
        ag.controller.wait(20)
        ag.controller.press("a")
        ag.controller.wait(50)
        _dshot("1_bag")
        ag.controller.navigate_menu(bag_index)  # move to the potion in the bag list + confirm
        ag.controller.wait(50)
        _dshot("2_sel")
        ag.controller.press("a")  # 'use on which POKéMON?' — the lead is first; select it
        ag.controller.wait(50)
        _dshot("3_afterA")
        ag.controller.mash_a(6, delay=30)  # clear the 'HP was restored' text, back to the battle
        ag.controller.wait(20)
        _dshot("4_done")
        heal_dbg["n"] += 1

    def _robust_flee():
        """Reliable menu-flee — used ONLY as a deadlock escape, not as normal play."""
        for _ in range(6):
            if not bt():
                return
            ag.controller.battle_menu_select("run")
            ag.controller.mash_a(3, delay=20)
            ag.controller.press("b")
            ag.controller.wait(15)

    def resolve_battle():
        """Resolve one encounter via the agent's heal-capable battle — FIGHT, never flee.

        We use ``run_battle_turn`` so the genome's move scoring AND ``hp_heal_threshold`` apply
        (the lead now carries potions, so healing is what keeps it alive through the grass). The
        caller sets ``hp_run_threshold`` very low, so the agent's flee branch effectively never
        fires — "stop fleeing". The only flee is a deadlock-escape: pokemon-kafka's stall guard
        makes ``run_battle_turn`` pick a weak "run" that can't end a wild battle (e.g. a Metapod
        Harden wall), freezing enemy HP; after a few frozen turns we flee robustly to break out.
        Trainers (type 2) can't be fled, so they resolve to a win (still in forest) or a faint.
        """
        bt0 = bt()
        last_enemy_hp = None
        stall = 0
        for _ in range(_BATTLE_TURN_CAP):
            if not bt():
                break
            battle = ag.memory.read_battle_state()
            hp_ratio = battle.player_hp / max(1, battle.player_max_hp)
            healing = ag.memory.find_healing_item()
            # HEAL ourselves when hurt (genome hp_heal_threshold) — pk's item action can't apply a
            # potion, so we drive the menu directly. The survival lever the broken item action hid.
            if hp_ratio < hp_heal and healing is not None:
                _heal_turn(healing[0])
                continue
            # Reset pk's wild stall counter so its choose_action doesn't lock into a weak "run".
            if hasattr(ag.battle_strategy, "_wild_fight_turns"):
                ag.battle_strategy._wild_fight_turns = 0
            enemy_hp = battle.enemy_hp
            stall = stall + 1 if (last_enemy_hp is not None and enemy_hp >= last_enemy_hp) else 0
            last_enemy_hp = enemy_hp
            if stall >= 8 and battle.battle_type == 1:
                _robust_flee()  # deadlock escape only (frozen wild, e.g. Harden wall)
                stall = 0
                continue
            ag.run_battle_turn()  # genome-scored fight (FIGHT branch works; only item is broken)
        post_map = ow()[0]
        won = post_map == FOREST  # survived the encounter (didn't white out)
        if bt0 == 2 and won:
            trainers["won"] += 1
        emit({"event_type": "battle_outcome", "turn": step["n"],
              "data": {"battle_type": bt0, "won": won}})
        return post_map

    def step_move(d):
        m, x, y, _ = ow()
        ag.controller.move(d)
        m2, x2, y2, _ = ow()
        if (x2, y2) == (x, y) and m2 == FOREST:
            ag.controller.move(d)  # first press only turned; press again to step
            m2, x2, y2, _ = ow()
        return m2 != FOREST or (x2, y2) != (x, y)

    note_map(ow()[0])  # emit the entrance map (beat 1)
    trace: list[tuple[int, int]] = []  # forest (x,y) visited, to characterize the wedge
    crossed = False
    stuck_in_battle = False

    route_tiles: list[tuple[int, int]] | None = None
    if route and Path(route).exists():
        route_tiles = [tuple(p) for p in json.loads(Path(route).read_text())["path"]]

    if route_tiles:
        path_index = {p: i for i, p in enumerate(route_tiles)}
        idx = 0
        blocked = 0
        while step["n"] < max_steps:
            step["n"] += 1
            if rec is not None:
                rec.tick(step["n"])
            if bt():
                post_map = resolve_battle()
                note_map(post_map)
                if bt():
                    stuck_in_battle = True
                    break
                continue
            m, x, y, tb = ow()
            note_map(m)
            if m == 47:  # north gate: climb column x=4 to y=1, sidle right to the door at x=5, up
                if y > 1 and x != 4:
                    ag.controller.move("left" if x > 4 else "right")
                elif y > 1:
                    ag.controller.move("up")
                elif x < 5:
                    ag.controller.move("right")
                else:
                    ag.controller.move("up")
                continue
            if m != FOREST:
                crossed = m in EXIT_MAPS
                break
            if tb:
                txt = ag.memory.read_dialogue()
                if "TRAINER" in txt.upper() or "TIPS" in txt.upper():
                    emit({"event_type": "discovery", "turn": step["n"],
                          "data": {"text": txt, "kind": "sign"}})
                ag.controller.mash_a(3, delay=15)
                continue
            if not trace or trace[-1] != (x, y):
                trace.append((x, y))
            ag.collision_map.update(ag.pyboy)
            ag.world.observe(FOREST, x, y, ag.collision_map.grid)
            pos = (x, y)
            if pos in path_index:
                idx = max(idx, path_index[pos])
            if idx + 1 >= len(route_tiles):
                ag.controller.move("up")  # at the exit approach — nudge onto the (1,0) warp
                continue
            if step_move(_dir_to(pos, route_tiles[idx + 1])):
                blocked = 0
                continue
            # Blocked. Two invisible traps live here (learned from the flood-fill ground truth):
            # a catcher's sight-line challenge dialogue locks movement with battle_type still 0
            # and text_box_active reading FALSE — mash A to advance it into the trainer battle;
            # and a plain-dialogue NPC standing on the tile — the A presses OPEN his text, and an
            # open box also freezes NPC wandering, so close it with B and wait him out.
            ag.controller.mash_a(3, delay=25)
            ag.controller.wait(15)
            if bt():
                continue
            ag.controller.press("b")
            ag.controller.wait(20)
            ag.controller.press("b")
            ag.controller.wait(80)
            blocked += 1
            if blocked >= 25:
                break  # genuinely wedged on the route

    for head, bias, cond in BEATS if route_tiles is None else []:
        blocked = 0
        sign_seen = False
        bag_at_beat = sum(q for _, q in ag.memory.read_bag_items())
        while step["n"] < max_steps:
            step["n"] += 1
            if rec is not None:
                rec.tick(step["n"])
            if bt():
                post_map = resolve_battle()
                note_map(post_map)
                if bt():  # battle never ended within the cap — bail the whole run
                    stuck_in_battle = True
                    break
                continue
            m, x, y, tb = ow()
            note_map(m)
            if m != FOREST:
                crossed = m in EXIT_MAPS
                break
            if tb:
                txt = ag.memory.read_dialogue()
                if "TRAINER" in txt.upper() or "TIPS" in txt.upper():
                    sign_seen = True
                    emit({"event_type": "discovery", "turn": step["n"],
                          "data": {"text": txt, "kind": "sign"}})
                ag.controller.mash_a(3, delay=15)
                continue
            if not trace or trace[-1] != (x, y):
                trace.append((x, y))
            ag.collision_map.update(ag.pyboy)
            ag.world.observe(FOREST, x, y, ag.collision_map.grid)
            # advance conditions
            if cond == "bag" and sum(q for _, q in ag.memory.read_bag_items()) > bag_at_beat:
                break
            if cond == "sign" and sign_seen:
                break
            if step_move(head):
                blocked = 0
                continue
            ag.controller.press("a")  # engage catcher/sign ahead
            ag.controller.wait(15)
            if bt():
                continue
            slid = False
            for s in [bias] + [p for p in PERP[head] if p != bias]:
                if step_move(s):
                    slid = True
                    break
            if not slid:
                blocked += 1
                if cond == "wall" and blocked >= 3:
                    break  # reached the wall/edge this beat aimed for
                if blocked >= 10:
                    break  # genuinely wedged on this beat
        if crossed or stuck_in_battle:
            break

    # Exit dash (legacy BEATS mode only): the BEATS get us into the far-left area but usually not
    # onto the warp. Greedily head for it — climb to the top edge (y->0), then sidle — with
    # wall-sliding and battle handling, until we leave the forest or run out of steps.
    target = (2, 0)
    dash_blocked = 0
    while route_tiles is None and not crossed and not stuck_in_battle and step["n"] < max_steps:
        step["n"] += 1
        if rec is not None:
            rec.tick(step["n"])
        if bt():
            post_map = resolve_battle()
            note_map(post_map)
            if bt():
                stuck_in_battle = True
                break
            continue
        m, x, y, tb = ow()
        note_map(m)
        if m != FOREST:
            crossed = m in EXIT_MAPS
            break
        if tb:
            ag.controller.mash_a(3, delay=15)
            continue
        if not trace or trace[-1] != (x, y):
            trace.append((x, y))
        ag.collision_map.update(ag.pyboy)
        ag.world.observe(FOREST, x, y, ag.collision_map.grid)
        if y > target[1]:
            primary = "up"  # climb to the top edge first
        elif x > target[0]:
            primary = "left"
        elif x < target[0]:
            primary = "right"
        else:
            primary = "up"  # at the warp column/row — nudge onto it
        if step_move(primary):
            dash_blocked = 0
            continue
        ag.controller.press("a")  # engage a catcher/sign blocking the way, then slide
        ag.controller.wait(15)
        if bt():
            continue
        if not any(step_move(s) for s in PERP[primary]):
            dash_blocked += 1
            if dash_blocked >= 15:
                break  # genuinely wedged approaching the exit

    m, x, y, _ = ow()
    note_map(m)
    crossed = m in EXIT_MAPS
    if out_state:
        with open(out_state, "wb") as f:
            ag.pyboy.save_state(f)
    if worldmap_out:
        ag.world.save(worldmap_out)
    if shot:
        ag.pyboy.tick(3, True)
        img = ag.pyboy.screen.image
        if img is not None:
            img.resize((img.width * 3, img.height * 3)).save(shot)

    from autotune.forest_story import score_forest
    from autotune.game_profile import detect_profile
    from autotune.party import OFF_LEVEL

    verdict = score_forest(events)
    xs = [p[0] for p in trace] or [x]
    ys = [p[1] for p in trace] or [y]
    fitness = {
        "turns": step["n"],
        "trainer_wins": trainers["won"],
        "lead_level": int(ag.pyboy.memory[detect_profile(ag.pyboy).party_base + OFF_LEVEL]),
        "final_map_id": m,
        "crossed": crossed,
        "stuck_in_battle": stuck_in_battle,
        "tiles_visited": len(trace),
        "x_range": [min(xs), max(xs)],
        "y_range": [min(ys), max(ys)],
        "reached_far_left": min(xs) <= 3,  # the exit pocket (warp at x=2) per forest-exit-wedge
    }
    if rec is not None:
        maps_visited = len(
            {e["data"]["map_id"] for e in events if e["event_type"] == "overworld"}
        )
        rec.finish({
            "turns": step["n"],
            "battles_won": trainers["won"],
            "maps_visited": maps_visited,
            "badges": 0,
            "reward": verdict.reward,
            "crossed": crossed,
            "furthest_beat_name": verdict.furthest_beat_name,
        })
    ag.pyboy.stop()
    return {
        "events": events,
        "trace": [list(p) for p in trace],
        "fitness": fitness,
        "reward": verdict.reward,
        "furthest_beat": verdict.furthest_beat,
        "furthest_beat_name": verdict.furthest_beat_name,
        "crossed": verdict.crossed,
        "trainer_wins": verdict.signals.trainer_wins,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genome-driven Viridian Forest follower.")
    parser.add_argument("--rom", required=True)
    parser.add_argument("--in-state", required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--worldmap-in", default=None)
    parser.add_argument("--worldmap-out", default=None)
    parser.add_argument("--out-state", default=None)
    parser.add_argument("--out", default=None, help="Write the result JSON here (else stdout).")
    parser.add_argument("--route", default=None,
                        help=f"Tile-path JSON to follow (default: {ROUTE_DEFAULT} if present; "
                             "pass --route '' to force the legacy BEATS nav).")
    parser.add_argument(
        "--record",
        default=None,
        help="Runs-dir to write a viewer-format run (frames + events) for the Paper Traces viewer.",
    )
    parser.add_argument("--label", default=None, help="Run label shown in the viewer.")
    parser.add_argument(
        "--frame-interval", type=int, default=10, help="Capture a frame every N steps."
    )
    args = parser.parse_args(argv)

    route = args.route
    if route is None and Path(ROUTE_DEFAULT).exists():
        route = ROUTE_DEFAULT

    genome = json.loads(os.environ.get("EVOLVE_PARAMS") or "{}")
    if not genome:
        from autotune.genome import base_genome

        genome = base_genome()

    result = follow_once(
        args.rom, args.in_state, genome,
        max_steps=args.max_steps,
        worldmap_in=args.worldmap_in,
        worldmap_out=args.worldmap_out,
        out_state=args.out_state,
        route=route or None,
        record_dir=args.record,
        label=args.label,
        frame_interval=args.frame_interval,
    )
    payload = {"genome": genome, **result}
    if args.out:
        Path(args.out).write_text(json.dumps(payload) + "\n")
    summary = {k: result[k] for k in ("reward", "furthest_beat_name", "crossed", "trainer_wins")}
    summary["turns"] = result["fitness"]["turns"]
    summary["lead_level"] = result["fitness"]["lead_level"]
    print(f"[forest_follow] {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
