"""Forest crossing v2: reuse PokemonAgent's battle handling (it FIGHTS trainers — the bug-catcher
gauntlet — and flees only wild), but drive the overworld toward the top-left exit (2,0) ourselves,
ignoring the quest logic that pulled the full agent backward. The v1 navigator fled everything and
got pinned in an un-fleeable trainer battle; this one beats the catchers."""

import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402
from world_map import WorldMap  # noqa: E402

ROM, IN_STATE, WM, OUT_STATE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 2500

FOREST = 51
EXIT = (2, 1)  # stand here, press up -> warp (2,0)
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)
ag.world = WorldMap.load(WM)

recent: list = []
seen = set()
crossed = False
pikachu_seen = False

for step in range(MAX):
    bt = ag.memory.read_battle_state()
    if getattr(bt, "battle_type", 0):
        if bt.enemy_species_name == "PIKACHU":
            pikachu_seen = True
            print(f"[nav2] step {step}: PIKACHU encountered (Lv{bt.enemy_level})")
        ag.run_battle_turn()  # fights trainers; flees only low-HP wild
        continue

    st = ag.memory.read_overworld_state()
    if getattr(st, "text_box_active", False):
        ag.controller.mash_a(3, delay=15)
        continue

    if st.map_id != FOREST:
        crossed = True
        with open(OUT_STATE, "wb") as f:
            ag.pyboy.save_state(f)
        print(f"[nav2] step {step}: CROSSED -> map {st.map_id} at ({st.x},{st.y}); saved {OUT_STATE}")
        break

    ag.collision_map.update(ag.pyboy)
    ag.world.observe(FOREST, st.x, st.y, ag.collision_map.grid)
    seen.add((st.x, st.y))
    here = (st.x, st.y)

    if here == EXIT:
        d = "up"
    elif ag.world.known_reachable(FOREST, st.x, st.y, *EXIT):
        d = ag.world.plan_step(FOREST, st.x, st.y, *EXIT, encounter_cost=4) or "up"
    else:
        d = ag.world.explore_step(FOREST, st.x, st.y) or ag.world.plan_step(FOREST, st.x, st.y, *EXIT) or "left"

    recent.append(here)
    recent = recent[-6:]
    if len(recent) == 6 and len(set(recent)) <= 2:
        d = {"up": "left", "down": "right", "left": "up", "right": "down"}.get(d, "left")

    before = (st.x, st.y)
    ag.controller.move(d)
    after = ag.memory.read_overworld_state()
    if after.map_id == FOREST and (after.x, after.y) == before:
        dx, dy = DELTA[d]
        ag.world.block(FOREST, before[0] + dx, before[1] + dy)

    if step % 20 == 0:
        kr = ag.world.known_reachable(FOREST, st.x, st.y, *EXIT)
        print(f"[nav2] step {step}: ({st.x},{st.y}) dir={d} seen={len(seen)} exit_reachable={kr}")

ag.world.save(WM)
print(f"[nav2] done: crossed={crossed} pikachu_seen={pikachu_seen} seen_tiles={len(seen)}")
ag.pyboy.stop()
sys.exit(0 if crossed else 2)
