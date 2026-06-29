"""Forest crossing v3: follow the full human route — up, right, up, then ALTERNATE left/up up the
top-left staircase to the exit warp at (2,0) — fighting the catcher at each turn. Hardened against
the v2 hang: a battle that doesn't progress (enemy HP flat for several turns, or too many turns) is
fled/broken instead of spun forever. Records the full trajectory for routes.json."""

import json
import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402

ROM, IN_STATE, OUT_STATE, TRACE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
FOREST = 51
LEGS = ["up", "right", "up"] + ["left", "up"] * 12  # then alternate to the top-left exit
STUCK_PER_LEG = 5
GLOBAL_STEP_CAP = 1200


def battle_type():
    return getattr(ag.memory.read_battle_state(), "battle_type", 0)


def resolve_battle():
    """Fight the current battle, but never spin: if enemy HP is flat for 6 turns or the fight runs
    long, flee/mash B to break out (a desynced menu or unwinnable spin)."""
    flat = 0
    last_hp = None
    for t in range(30):
        if not battle_type():
            return
        bs = ag.memory.read_battle_state()
        hp = getattr(bs, "enemy_hp", None)
        if last_hp is not None and hp is not None and hp >= last_hp:
            flat += 1
        else:
            flat = 0
        last_hp = hp
        if flat >= 6 or t >= 24:  # stuck or dragging — bail out
            ag.controller.battle_menu_select("run")
            ag.controller.mash_a(4, delay=20)
            for _ in range(4):
                ag.controller.press("b")
                ag.controller.wait(15)
            return
        ag.run_battle_turn()


ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)

trace = []
step = 0
crossed = False


def ow():
    s = ag.memory.read_overworld_state()
    return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)


for leg in LEGS:
    stuck = 0
    while stuck < STUCK_PER_LEG and step < GLOBAL_STEP_CAP:
        step += 1
        if battle_type():
            resolve_battle()
            continue
        m, x, y, tb = ow()
        if m != FOREST:
            crossed = True
            break
        if tb:
            ag.controller.mash_a(3, delay=15)
            continue
        if not trace or (trace[-1]["x"], trace[-1]["y"]) != (x, y):
            trace.append({"x": x, "y": y, "leg": leg, "step": step})
        before = (x, y)
        ag.controller.move(leg)
        m2, x2, y2, _ = ow()
        if m2 != FOREST:
            crossed = True
            break
        if (x2, y2) == before:
            ag.controller.press("a")  # engage catcher / read sign
            ag.controller.wait(20)
            stuck += 1
        else:
            stuck = 0
    if crossed:
        break

m, x, y, _ = ow()
crossed = m != FOREST
# Always save the final state (wedge or crossing) so we can inspect/resume from it.
with open(OUT_STATE, "wb") as f:
    ag.pyboy.save_state(f)

# Dump the collision grid + sprite slots around where we ended, to SEE the maze at the wedge.
ag.collision_map.update(ag.pyboy)
grid = [row[:] for row in ag.collision_map.grid]
sprites = []
for i in range(16):
    base = 0xC100 + i * 0x10
    sy, sx = ag.pyboy.memory[base + 4], ag.pyboy.memory[base + 6]
    if sx or sy:
        sprites.append({"slot": i, "sx": sx, "sy": sy})
json.dump({"crossed": crossed, "final": {"map": m, "x": x, "y": y}, "steps": step,
           "lead_level": ag.pyboy.memory[0xD18C], "trace": trace,
           "collision": grid, "sprites": sprites}, open(TRACE, "w"))
print(f"[route3] crossed={crossed} final=map{m}({x},{y}) steps={step} leadL={ag.pyboy.memory[0xD18C]} pts={len(trace)}")
ag.pyboy.stop()
sys.exit(0 if crossed else 2)
