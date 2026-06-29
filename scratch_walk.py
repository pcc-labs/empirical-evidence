"""Walk one direction N tiles, FIGHTING every encounter (grass + catchers) for leveling/discovery,
clearing text, engaging a blocking catcher with A. Saves the resulting state + a screenshot so we
can step the forest leg-by-leg. Hang-guarded: a battle that stalls is fled.

Usage: scratch_walk.py ROM IN_STATE OUT_STATE SHOT DIR NTILES
"""

import sys

sys.path.insert(0, "scripts")
from agent import PokemonAgent  # noqa: E402

ROM, IN_STATE, OUT_STATE, SHOT, DIR = sys.argv[1:6]
NTILES = int(sys.argv[6]) if len(sys.argv) > 6 else 8
FOREST = 51

ag = PokemonAgent(ROM, strategy="low")
with open(IN_STATE, "rb") as f:
    ag.pyboy.load_state(f)


def bt():
    return getattr(ag.memory.read_battle_state(), "battle_type", 0)


def resolve_battle():
    flat, last = 0, None
    for t in range(30):
        if not bt():
            return
        hp = getattr(ag.memory.read_battle_state(), "enemy_hp", None)
        flat = flat + 1 if (last is not None and hp is not None and hp >= last) else 0
        last = hp
        if flat >= 6 or t >= 24:
            ag.controller.battle_menu_select("run")
            ag.controller.mash_a(4, delay=20)
            for _ in range(4):
                ag.controller.press("b"); ag.controller.wait(15)
            return
        ag.run_battle_turn()


def ow():
    s = ag.memory.read_overworld_state()
    return s.map_id, s.x, s.y, getattr(s, "text_box_active", False)


advanced = 0
blocked = 0
battles = 0
lvl0 = ag.pyboy.memory[0xD18C]
guard = 0
while advanced < NTILES and guard < NTILES * 20 + 60:
    guard += 1
    if bt():
        battles += 1
        resolve_battle()
        continue
    m, x, y, tb = ow()
    if m != FOREST:
        break
    if tb:
        ag.controller.mash_a(3, delay=15)
        continue
    before = (x, y)
    ag.controller.move(DIR)
    m2, x2, y2, _ = ow()
    if m2 != FOREST:
        break
    if (x2, y2) == before:
        # The first press after a direction change only TURNS in place (no step). Press again to
        # actually move before deciding it's blocked.
        ag.controller.move(DIR)
        m2, x2, y2, _ = ow()
        if m2 != FOREST:
            break
    if (x2, y2) == before:
        ag.controller.press("a")  # genuinely blocked — engage catcher / read sign
        ag.controller.wait(20)
        if bt():
            continue
        blocked += 1
        if blocked >= 4:
            break
    else:
        advanced += 1
        blocked = 0

with open(OUT_STATE, "wb") as f:
    ag.pyboy.save_state(f)
ag.pyboy.tick(3, True)
img = ag.pyboy.screen.image
if img is not None:
    img.resize((img.width * 3, img.height * 3)).save(SHOT)

m, x, y, _ = ow()
lvl = ag.pyboy.memory[0xD18C]
print(f"[walk] dir={DIR} advanced={advanced}/{NTILES} blocked={blocked} battles={battles} "
      f"-> map={m} pos=({x},{y}) L{lvl0}->L{lvl}")
ag.pyboy.stop()
