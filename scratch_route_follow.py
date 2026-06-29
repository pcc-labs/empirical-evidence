"""Follow a human-described route through Viridian Forest and RECORD the real coordinate trajectory,
so we can write it into routes.json as observed waypoints.

Route (from the user playing it): up -> right (all the way) -> up -> left -> up, with a bug-catcher
trainer at each turn that you engage by talking (pressing A toward them). Reuses PokemonAgent's
battle handling (it fights trainers; flees only low-HP wild). Logs every distinct tile + each
turn-point (where a leg ends) and each trainer battle, then saves the post-forest state on crossing.
"""

import json
import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402

ROM, IN_STATE, OUT_STATE, TRACE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
LEGS = ["up", "right", "up", "left", "up"]
FOREST = 51
PER_LEG_LIMIT = 60  # max steps per leg before giving up on it

ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)

trace = []  # list of {step, x, y, event}
turns = []  # turn-points (waypoints)
step = 0
crossed = False


def pos():
    s = ag.memory.read_overworld_state()
    return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)


def clear_battles_and_text():
    """If in a battle, fight it out; if a text box is up, advance it. Returns True if it acted."""
    bt = ag.memory.read_battle_state()
    if getattr(bt, "battle_type", 0):
        sp = bt.enemy_species_name
        trace.append({"step": step, "event": f"battle:{sp} L{bt.enemy_level} type{bt.battle_type}"})
        for _ in range(40):  # resolve the whole battle
            if not getattr(ag.memory.read_battle_state(), "battle_type", 0):
                break
            ag.run_battle_turn()
        return True
    m, x, y, tb = pos()
    if tb:
        ag.controller.mash_a(3, delay=15)
        return True
    return False


for leg in LEGS:
    last = None
    stuck = 0
    while stuck < 6 and step < PER_LEG_LIMIT * len(LEGS):
        step += 1
        if clear_battles_and_text():
            continue
        m, x, y, tb = pos()
        if m != FOREST:
            crossed = True
            break
        if (x, y) != last:
            trace.append({"step": step, "x": x, "y": y, "leg": leg})
            last = (x, y)
        before = (x, y)
        ag.controller.move(leg)
        m2, x2, y2, _ = pos()
        if (x2, y2) == before and m2 == FOREST:
            # blocked — maybe a catcher is in the way; talk to engage, else count as stuck
            ag.controller.press("a")
            ag.controller.wait(20)
            if getattr(ag.memory.read_battle_state(), "battle_type", 0):
                continue  # battle started; next loop fights it
            stuck += 1
    if crossed:
        break
    m, x, y, _ = pos()
    turns.append({"leg_end": leg, "x": x, "y": y})

m, x, y, _ = pos()
if m != FOREST:
    crossed = True
    with open(OUT_STATE, "wb") as f:
        ag.pyboy.save_state(f)

json.dump({"crossed": crossed, "final": {"map": m, "x": x, "y": y},
           "turns": turns, "trace": trace,
           "lead_level": ag.pyboy.memory[0xD18C]}, open(TRACE, "w"), indent=1)
print(f"[route] crossed={crossed} final=map{m}({x},{y}) turns={turns} leadL={ag.pyboy.memory[0xD18C]}")
ag.pyboy.stop()
sys.exit(0 if crossed else 2)
