"""Forest crossing v2: user's opening legs (up,right,up,left,up) THEN goal-directed walk toward the
top-left exit (2,1) -> step up into the warp (2,0). Turns at walls, presses A on a block (engage a
catcher / read a sign), fights battles via PokemonAgent. Records the full trajectory so the crossing
route can be written into routes.json. Anti-oscillation: avoid immediately reversing."""

import json
import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402

ROM, IN_STATE, OUT_STATE, TRACE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BUDGET = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
FOREST = 51
EXIT = (2, 1)
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)

trace = []
crossed = False
last_dir = None
blocked_here = 0


def st():
    s = ag.memory.read_overworld_state()
    return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)


def fight_or_text():
    bt = ag.memory.read_battle_state()
    if getattr(bt, "battle_type", 0):
        for _ in range(40):
            if not getattr(ag.memory.read_battle_state(), "battle_type", 0):
                break
            ag.run_battle_turn()
        return True
    if st()[3]:
        ag.controller.mash_a(3, delay=15)
        return True
    return False


def goal_dirs(x, y):
    """Preferred directions toward the exit: bigger remaining axis first."""
    dx, dy = x - EXIT[0], y - EXIT[1]
    horiz = "left" if dx > 0 else ("right" if dx < 0 else None)
    vert = "up" if dy > 0 else ("down" if dy < 0 else None)
    order = [horiz, vert] if abs(dx) >= abs(dy) else [vert, horiz]
    return [d for d in order if d]


step = 0
# Phase 1: the user's opening legs through the catcher section.
phase1 = ["up", "right", "up", "left", "up"]
for leg in phase1:
    stuck = 0
    while stuck < 5 and step < BUDGET:
        step += 1
        if fight_or_text():
            continue
        m, x, y, _ = st()
        if m != FOREST:
            crossed = True
            break
        trace.append({"x": x, "y": y, "phase": 1, "leg": leg})
        before = (x, y)
        ag.controller.move(leg)
        m2, x2, y2, _ = st()
        if m2 == FOREST and (x2, y2) == before:
            ag.controller.press("a")
            ag.controller.wait(20)
            stuck += 1
        else:
            stuck = 0
    if crossed:
        break

# Phase 2: goal-directed toward the exit, turning at walls, talking on block.
while not crossed and step < BUDGET:
    step += 1
    if fight_or_text():
        continue
    m, x, y, _ = st()
    if m != FOREST:
        crossed = True
        break
    if (x, y) == EXIT:
        cand = ["up"]
    else:
        cand = goal_dirs(x, y)
        # try goal dirs, then the rest, but avoid reversing unless nothing else
        rest = [d for d in ("up", "left", "down", "right") if d not in cand]
        cand = cand + rest
        if last_dir and OPP.get(last_dir) in cand and len(cand) > 1:
            cand = [d for d in cand if d != OPP[last_dir]] + [OPP[last_dir]]
    trace.append({"x": x, "y": y, "phase": 2})
    moved = False
    for d in cand:
        before = (x, y)
        ag.controller.move(d)
        m2, x2, y2, _ = st()
        if m2 != FOREST:
            crossed = True
            moved = True
            break
        if (x2, y2) != before:
            last_dir = d
            moved = True
            blocked_here = 0
            break
    if not moved:
        # fully blocked — talk (catcher/sign), then allow any dir next loop
        ag.controller.press("a")
        ag.controller.wait(20)
        blocked_here += 1
        last_dir = None
        if blocked_here > 8:
            break  # genuinely wedged

m, x, y, _ = st()
if m != FOREST:
    crossed = True
    with open(OUT_STATE, "wb") as f:
        ag.pyboy.save_state(f)

json.dump({"crossed": crossed, "final": {"map": m, "x": x, "y": y}, "steps": step,
           "trace": trace, "lead_level": ag.pyboy.memory[0xD18C]}, open(TRACE, "w"))
print(f"[route2] crossed={crossed} final=map{m}({x},{y}) steps={step} trace={len(trace)}")
ag.pyboy.stop()
sys.exit(0 if crossed else 2)
