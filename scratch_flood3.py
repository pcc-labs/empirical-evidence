"""Forest flood-fill v3 — clean_flood + floodfill2, hardened against the (26,19) livelock.

Combines the proven pieces and fixes the two ways the earlier fills wedged:
  - retry-before-wall (3x) so ~4% flaky presses never seal a junction (floodfill2);
  - frontier-path backtrack: BFS over VISITED tiles to the nearest tile bordering unexplored
    space, instead of single-step parent walk (floodfill2);
  - NEW: text-box clearing — a post-battle/sign dialogue disables all movement; clean_flood
    never mashed it away, so the DFS spun at one tile forever (the (26,19) freeze);
  - NEW: broken-edge replanning — if a step along a backtrack path fails, the visited->visited
    edge is recorded broken and BFS routes around it (never retries the same edge forever).

Usage: scratch_flood3.py ROM STATE OUT_STATE DUMP_JSON [max_iters]
"""

import json
import sys
from collections import deque

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402

ROM, STATE, OUT_STATE, DUMP = sys.argv[1:5]
MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
FOREST = 51
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ORDER = ["up", "left", "right", "down"]  # exit is top-left: bias up, then left

ag = PokemonAgent(ROM, strategy="low")
with open(STATE, "rb") as f:
    ag.pyboy.load_state(f)
mem = ag.pyboy.memory


def ow():
    return ag.memory.read_overworld_state()


def in_battle():
    return ag.memory.read_battle_state().battle_type != 0


def fight():
    for _ in range(15):
        if not in_battle():
            return
        ag.controller.battle_menu_select("fight")
        ag.controller.navigate_menu(0)
        ag.controller.wait(120)
        ag.controller.mash_a(6, delay=25)


def clear_tb():
    """Mash away any modal text box (post-battle text, sign, pickup) — it blocks ALL movement."""
    for _ in range(8):
        if not getattr(ow(), "text_box_active", False):
            return True
        ag.controller.mash_a(2, delay=15)
        ag.controller.press("b")
        ag.controller.wait(10)
    return not getattr(ow(), "text_box_active", False)


def heal():
    n = mem[0xD163]
    for i in range(min(n, 6)):
        b = 0xD16B + i * 44
        mx = (mem[b + 34] << 8) | mem[b + 35]
        if mx:
            mem[b + 1] = mx >> 8
            mem[b + 2] = mx & 0xFF


def dir_to(a, b):
    if b[0] > a[0]:
        return "right"
    if b[0] < a[0]:
        return "left"
    if b[1] > a[1]:
        return "down"
    return "up"


def try_move(d, tries=3):
    """One tile in dir d with retry; clears battles and text boxes. Returns (x, y).

    A failed press may mean a bug catcher's sight-line fired: his challenge dialogue locks ALL
    movement, battle_type is still 0, and text_box_active reads FALSE for it (the (26,19) trap
    every explorer died in). So after each failed press, mash A — it advances any hidden
    dialogue into the trainer battle, which fight() then wins.
    """
    cur = (ow().x, ow().y)
    for _ in range(tries):
        if getattr(ow(), "text_box_active", False):
            clear_tb()
        ag.controller.move(d)
        if in_battle():
            ag.controller.mash_a(4, delay=20)
            fight()
            clear_tb()
            return (ow().x, ow().y)
        if getattr(ow(), "text_box_active", False):
            clear_tb()
        a = (ow().x, ow().y)
        if a != cur or ow().map_id != FOREST:
            return a
        ag.controller.mash_a(3, delay=25)  # advance a hidden challenge dialogue / engage ahead
        ag.controller.wait(20)
        if in_battle():
            fight()
            clear_tb()
            return (ow().x, ow().y)
        # a plain-dialogue NPC: the A presses OPEN his text (text_box_active lies) and an open
        # box also freezes NPC wandering — close it unconditionally and give him time to move
        ag.controller.press("b")
        ag.controller.wait(20)
        ag.controller.press("b")
        ag.controller.wait(80)
    return cur  # failed `tries` clean presses -> wall


# South-gate warp mats: never step back onto them (stepping on one warps out the bottom).
# The start state stands ON one; wall it as soon as we step off. North exit (1,0)/(2,0) is the goal.
SOUTH_MATS = {(15, 47), (16, 47), (17, 47), (18, 47)}
start = (ow().x, ow().y)
visited = {start}
walls = SOUTH_MATS - {start}
start_walled = start not in SOUTH_MATS
broken = set()  # visited->visited edges that failed during backtrack: {(a, b), ...}
recoveries = 0
miny = start[1]
crossed = False
sealed = False
livelock = 0
last_sig = None
it = 0


def path_to_frontier(cur):
    """BFS over visited tiles (skipping broken edges) to the nearest tile bordering unknown."""
    prev = {cur: None}
    q = deque([cur])
    while q:
        c = q.popleft()
        if any(
            (c[0] + dx, c[1] + dy) not in visited and (c[0] + dx, c[1] + dy) not in walls
            for dx, dy in DIRS.values()
        ) and c != cur:
            path = [c]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return list(reversed(path))[1:]
        for dx, dy in DIRS.values():
            nb = (c[0] + dx, c[1] + dy)
            if nb in visited and nb not in prev and (c, nb) not in broken:
                prev[nb] = c
                q.append(nb)
    return None


for it in range(MAX):
    heal()
    if it and it % 1000 == 0:
        s = ow()
        print(f"  [tick] it {it}: at ({s.x},{s.y}) visited={len(visited)} walls={len(walls)} "
              f"broken={len(broken)} miny={miny}", flush=True)
    s = ow()
    sig = (s.map_id, s.x, s.y, len(visited), len(walls), len(broken))
    livelock = livelock + 1 if sig == last_sig else 0
    last_sig = sig
    if livelock >= 300:
        print(f"*** LIVELOCK at map {s.map_id} ({s.x},{s.y}) it {it} — aborting ***", flush=True)
        break
    if s.map_id != FOREST:
        if it % 50 == 0:
            print(f"*** off forest it {it}: map {s.map_id} ({s.x},{s.y}) ***", flush=True)
        if s.map_id in (47, 2):
            print("*** CROSSED FOREST (north gate) ***", flush=True)
            crossed = True
            break
        if s.map_id == 13:  # Route 2 is one map: north segment = crossed, south = bounced out
            crossed = s.y <= 30
            print(f"*** on Route 2 at y={s.y} -> crossed={crossed} ***", flush=True)
            break
        # bounced into the south gate building (50): step off the arrival mat, then back on
        clear_tb()
        ag.controller.move("down" if it % 2 == 0 else "up")
        continue
    if getattr(s, "text_box_active", False):
        clear_tb()
        continue
    cur = (s.x, s.y)
    if not start_walled and cur != start:
        walls.add(start)  # off the entry mat now — never route back onto it
        visited.discard(start)
        start_walled = True
    if cur not in visited and cur not in walls:
        visited.add(cur)  # battles/pushback can land us somewhere new — absorb it
    if cur[1] < miny:
        miny = cur[1]
        print(f"  it {it}: min y={miny} at {cur} visited={len(visited)} walls={len(walls)}",
              flush=True)
    adv = False
    for d in ORDER:
        dx, dy = DIRS[d]
        nb = (cur[0] + dx, cur[1] + dy)
        if nb in visited or nb in walls:
            continue
        a = try_move(d)
        if ow().map_id != FOREST:
            adv = True
            break
        if a == cur:
            walls.add(nb)
        else:
            visited.add(a)
            adv = True
            break
    if adv:
        continue
    path = path_to_frontier(cur)
    if path is None:
        has_frontier = any(
            (vx + dx, vy + dy) not in visited and (vx + dx, vy + dy) not in walls
            for vx, vy in visited for dx, dy in DIRS.values()
        )
        if has_frontier and broken and recoveries < 30:
            # broken edges (transient blockers: NPC in the way, mid-animation press) have cut us
            # off from the frontier — forget them, wait a beat, and replan
            recoveries += 1
            print(f"  [recover {recoveries}] clearing {len(broken)} broken edges at {cur}",
                  flush=True)
            broken.clear()
            ag.controller.press("b")
            ag.controller.wait(20)
            ag.controller.press("b")
            ag.controller.wait(240)  # close any lingering dialogue; let a blocking NPC wander off
            continue
        print(f"  FULLY EXPLORED: {len(visited)} tiles, min y={miny} "
              f"(frontier remains: {has_frontier})", flush=True)
        sealed = True
        break
    for nxt in path:
        here = (ow().x, ow().y)
        if here == nxt:
            continue
        a = try_move(dir_to(here, nxt))
        if ow().map_id != FOREST:
            break
        if a != nxt:  # backtrack edge failed -> mark broken, replan next iteration
            broken.add((here, nxt))
            print(f"  [broken] {here}->{nxt} it {it}", flush=True)
            break

with open(OUT_STATE, "wb") as f:
    ag.pyboy.save_state(f)
json.dump({"visited": sorted(visited), "walls": sorted(walls),
           "broken": sorted(broken)}, open(DUMP, "w"))
ys = [y for _x, y in visited]
frontier = sorted(
    nb for vx, vy in visited for dx, dy in DIRS.values()
    for nb in [(vx + dx, vy + dy)]
    if nb not in visited and nb not in walls and nb[0] >= 0 and nb[1] >= 0
)
print(f"\nDONE crossed={crossed} sealed={sealed} iters={it + 1}/{MAX} visited={len(visited)} "
      f"y {min(ys)}-{max(ys)} walls={len(walls)} broken={len(broken)} "
      f"frontier={len(set(frontier))}", flush=True)
exitn = [(2, 0), (1, 0), (2, 1), (1, 1)]
print("exit tiles visited:", {t: (t in visited) for t in exitn})
for row in range(4):
    vis = sorted(x for x, y in visited if y == row)
    wal = sorted(x for x, y in walls if y == row)
    print(f"  row y={row}: visited x={vis} walls x={wal}", flush=True)
print("frontier (y<=5):", sorted(t for t in set(frontier) if t[1] <= 5), flush=True)
ag.pyboy.stop()
