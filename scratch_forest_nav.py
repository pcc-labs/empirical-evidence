"""Focused Viridian-Forest crossing navigator (one-off, to capture a post-forest state).

Strips the confounds that sent the full agent backward (quest targeting, force-fight whiteouts):
ONLY navigates map 51 toward the north warp at (2,0) — reachable by standing on (2,1) and pressing
up — flees every wild battle, and explores around the x=4-5 wall into the exit pocket. Instrumented
so we see exactly where it goes. On crossing (map_id leaves 51) it saves a state and stops.
"""

import sys
import time

sys.path.insert(0, "scripts")  # pokemon-kafka/scripts

from agent import GameController  # noqa: E402
from memory_reader import CollisionMap, MemoryReader  # noqa: E402
from pyboy import PyBoy  # noqa: E402
from world_map import WorldMap  # noqa: E402

ROM = sys.argv[1]
IN_STATE = sys.argv[2]
WM_FILE = sys.argv[3]
OUT_STATE = sys.argv[4]
MAX_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000

FOREST = 51
EXIT_BELOW = (2, 1)  # stand here, press up -> warp at (2,0)

pb = PyBoy(ROM, window="null")
with open(IN_STATE, "rb") as f:
    pb.load_state(f)
mem = MemoryReader(pb)
ctrl = GameController(pb)
cmap = CollisionMap()
world = WorldMap.load(WM_FILE)

recent: list[tuple[int, int]] = []
seen_tiles: set[tuple[int, int]] = set()
crossed = False
t0 = time.time()

for step in range(MAX_STEPS):
    # 1) flee any battle
    bt = mem.read_battle_state()
    if getattr(bt, "battle_type", 0):
        ctrl.battle_menu_select("run")
        ctrl.mash_a(6, delay=20)
        continue

    st = mem.read_overworld_state()
    if getattr(st, "text_box_active", False):
        ctrl.mash_a(4, delay=15)
        continue

    # 2) crossed?
    if st.map_id != FOREST:
        crossed = True
        with open(OUT_STATE, "wb") as f:
            pb.save_state(f)
        print(f"[nav] step {step}: CROSSED forest -> map {st.map_id} at ({st.x},{st.y}); saved {OUT_STATE}")
        break

    # 3) observe surroundings into the world map
    cmap.update(pb)
    world.observe(FOREST, st.x, st.y, cmap.grid)
    seen_tiles.add((st.x, st.y))

    here = (st.x, st.y)
    # 4) at the exit-approach tile -> step up into the warp
    if here == EXIT_BELOW:
        direction = "up"
    elif world.known_reachable(FOREST, st.x, st.y, *EXIT_BELOW):
        direction = world.plan_step(FOREST, st.x, st.y, *EXIT_BELOW, encounter_cost=6) or "up"
    else:
        # exit pocket not yet known-reachable -> map new ground to find the way around the wall
        direction = world.explore_step(FOREST, st.x, st.y)
        if direction is None:
            direction = world.plan_step(FOREST, st.x, st.y, *EXIT_BELOW) or "left"

    # 5) oscillation breaker: if stuck rattling between the same couple tiles, nudge perpendicular
    recent.append(here)
    recent = recent[-6:]
    if len(recent) == 6 and len(set(recent)) <= 2:
        direction = {"up": "left", "down": "right", "left": "up", "right": "down"}.get(direction, "left")

    before = (st.x, st.y)
    ctrl.move(direction)
    after_st = mem.read_overworld_state()
    if after_st.map_id == FOREST and (after_st.x, after_st.y) == before:
        # the step failed -> that tile is a wall the collision grid lied about; hard-block it
        dx = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
        world.block(FOREST, before[0] + dx[0], before[1] + dx[1])

    if step % 25 == 0:
        kr = world.known_reachable(FOREST, st.x, st.y, *EXIT_BELOW)
        print(f"[nav] step {step}: ({st.x},{st.y}) dir={direction} seen={len(seen_tiles)} exit_reachable={kr}")

world.save(WM_FILE)
print(f"[nav] done: crossed={crossed} steps<= {MAX_STEPS} seen_tiles={len(seen_tiles)} "
      f"elapsed={time.time() - t0:.0f}s")
pb.stop()
sys.exit(0 if crossed else 2)
