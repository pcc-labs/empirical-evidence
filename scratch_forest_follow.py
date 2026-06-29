"""Landmark-follower for Viridian Forest, driven by the human walkthrough (general directions +
OBSERVED landmarks, never coordinates). It:

  - drives a sequence of headings from the walkthrough, wall-following around trees (slide
    perpendicular when the heading is blocked, then resume the heading);
  - FIGHTS every encounter (wild + trainers) for leveling and discovery (per user: catchers are
    discovery data, never avoided);
  - advances each beat on an OBSERVATION: bag item count +1, trainers-defeated count, the
    "Trainer Tips" sign text (via memory_reader.read_dialogue, merged in #32), a wall/edge, or a
    map_id change;
  - PERSISTS the WorldMap and records the full tile path, so progress is learned, not thrown away.

This is the scaffold whose crossings become the forest story/reward for training weights.

Usage: scratch_forest_follow.py ROM IN_STATE OUT_STATE WM TRACE SHOT [MAXSTEPS]
"""

import json
import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402
from world_map import WorldMap  # noqa: E402

ROM, IN_STATE, OUT_STATE, WM, TRACE, SHOT = sys.argv[1:7]
MAXSTEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 1500
FOREST = 51
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
PERP = {"up": ["left", "right"], "down": ["left", "right"],
        "left": ["up", "down"], "right": ["up", "down"]}

# Beats from the walkthrough: (heading, slide-bias toward this perpendicular first, advance-when).
# advance: "wall" = blocked after sliding; "bag" = bag count increased; "sign" = Trainer Tips read;
# "map" = left the forest. Headings are the general direction; wall-following handles the trees.
BEATS = [
    ("left",  "up",   "wall"),   # 1a: walk left from entrance to the wall
    ("up",    "left", "wall"),   # 1b: head north
    ("left",  "up",   "wall"),   # 1c: turn left again
    ("up",    "right","bag"),    # 2+3: head north up the path -> pick up Poke Ball
    ("up",    "left", "bag"),    # 4+5: past trainers, stick LEFT -> Antidote
    ("left",  "down", "wall"),   # 6: into the grassy area, loop
    ("up",    "left", "sign"),   # 7a: north until the Trainer Tips sign
    ("left",  "up",   "wall"),   # 8a: head west to the far-left edge
    ("up",    "left", "map"),    # 8b: walk north -> exit gate to Route 2
]

ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)
try:
    ag.world = WorldMap.load(WM)
except Exception:
    pass

trace = []
beats_done = []
trainers = 0


def bt():
    return getattr(ag.memory.read_battle_state(), "battle_type", 0)


def flee():
    for _ in range(6):
        if not bt():
            return
        ag.controller.battle_menu_select("run")
        ag.controller.mash_a(3, delay=20)
        ag.controller.press("b"); ag.controller.wait(15)


def resolve_battle():
    """Fight CATCHERS (trainers) — they're discovery data and can't be fled — but FLEE wild grass
    encounters. Fighting every wild battle with an empty bag chips even an L13 lead to a faint over
    100+ grass tiles (observed: whiteout to Pallet). Fleeing wild preserves HP to actually cross."""
    global trainers
    if ag.memory.read_battle_state().battle_type == 1:  # wild
        flee()
        return
    flat, last = 0, None  # trainer: fight it
    for t in range(30):
        if not bt():
            trainers += 1
            return
        hp = getattr(ag.memory.read_battle_state(), "enemy_hp", None)
        flat = flat + 1 if (last is not None and hp is not None and hp >= last) else 0
        last = hp
        if flat >= 8 or t >= 28:
            flee()
            return
        ag.run_battle_turn()


def ow():
    s = ag.memory.read_overworld_state()
    return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)


def step_move(d):
    """Try one tile in direction d (turn, then step). Return True if position changed."""
    m, x, y, _ = ow()
    ag.controller.move(d)
    m2, x2, y2, _ = ow()
    if (x2, y2) == (x, y) and m2 == FOREST:
        ag.controller.move(d)  # first press only turned; press again to step
        m2, x2, y2, _ = ow()
    return m2 != FOREST or (x2, y2) != (x, y)


def bag_count():
    return sum(q for _, q in ag.memory.read_bag_items())


step = 0
bag0 = bag_count()
crossed = False
for bi, (head, bias, cond) in enumerate(BEATS):
    blocked = 0
    bag_at_beat = bag_count()
    sign_seen = False
    while step < MAXSTEPS:
        step += 1
        if bt():
            resolve_battle()
            continue
        m, x, y, tb = ow()
        if m != FOREST:
            crossed = True
            break
        if tb:
            txt = ag.memory.read_dialogue()
            if "TRAINER" in txt.upper() or "TIPS" in txt.upper():
                sign_seen = True
            ag.controller.mash_a(3, delay=15)
            continue
        ag.collision_map.update(ag.pyboy)
        ag.world.observe(FOREST, x, y, ag.collision_map.grid)
        if not trace or (trace[-1]["x"], trace[-1]["y"]) != (x, y):
            trace.append({"x": x, "y": y, "beat": bi, "head": head})
        # advance conditions
        if cond == "bag" and bag_count() > bag_at_beat:
            break
        if cond == "sign" and sign_seen:
            break
        # drive heading; if blocked, slide along the bias/perpendicular, then resume
        if step_move(head):
            blocked = 0
            continue
        # blocked on heading -> engage whatever's ahead (catcher/sign), then slide
        ag.controller.press("a"); ag.controller.wait(15)
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
    beats_done.append({"beat": bi, "head": head, "cond": cond,
                       "done_at": ow()[1:3], "step": step, "crossed_here": crossed})
    if crossed:
        break

m, x, y, _ = ow()
crossed = m != FOREST
with open(OUT_STATE, "wb") as f:
    ag.pyboy.save_state(f)
ag.world.save(WM)
ag.pyboy.tick(3, True)
img = ag.pyboy.screen.image
if img is not None:
    img.resize((img.width * 3, img.height * 3)).save(SHOT)
json.dump({"crossed": crossed, "final": {"map": m, "x": x, "y": y}, "steps": step,
           "lead_level": ag.pyboy.memory[0xD18C], "trainers": trainers,
           "bag_gained": bag_count() - bag0, "beats": beats_done,
           "trace_points": len(trace), "trace": trace}, open(TRACE, "w"))
print(f"[follow] crossed={crossed} final=map{m}({x},{y}) steps={step} "
      f"L{ag.pyboy.memory[0xD18C]} trainers={trainers} bag+={bag_count()-bag0} "
      f"beats_reached={len(beats_done)}/{len(BEATS)} tiles={len(trace)}")
ag.pyboy.stop()
sys.exit(0 if crossed else 2)
