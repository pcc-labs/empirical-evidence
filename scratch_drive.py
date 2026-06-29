"""Screenshot-driven manual controller for PyBoy — for hand-crossing Viridian Forest.

Usage:
  python scratch_drive.py ROM ROLLING_STATE SHOT.png MOVES [--init INIT_STATE]

MOVES is comma-separated button presses: up/down/left/right/a/b/start/select, or "wait".
Loads ROLLING_STATE (or INIT_STATE the first time with --init), applies the moves, saves the
rolling state back, writes an upscaled screenshot, and prints map/pos/battle/party so we can see
where we are and decide the next batch.
"""

import os
import sys

sys.path.insert(0, "scripts")
from agent import GameController  # noqa: E402
from memory_reader import MemoryReader  # noqa: E402
from pyboy import PyBoy  # noqa: E402

ROM, ROLLING, SHOT, MOVES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
INIT = None
if "--init" in sys.argv:
    INIT = sys.argv[sys.argv.index("--init") + 1]

pb = PyBoy(ROM, window="null")
load_from = ROLLING if os.path.exists(ROLLING) else INIT
if load_from:
    with open(load_from, "rb") as f:
        pb.load_state(f)
else:
    print(f"ERROR: no rolling state {ROLLING} and no --init")
    sys.exit(1)

ctrl = GameController(pb)
mem = MemoryReader(pb)

moves = [m.strip() for m in MOVES.split(",") if m.strip()] if MOVES else []
for mv in moves:
    if mv in ("up", "down", "left", "right"):
        ctrl.move(mv)
    elif mv == "wait":
        ctrl.wait(40)
    else:  # a, b, start, select
        ctrl.press(mv)
        ctrl.wait(20)

with open(ROLLING, "wb") as f:
    pb.save_state(f)

# screenshot (upscale x3 for readability). Force a rendered frame first — in null-window mode the
# framebuffer can be left half-composited after tick-without-render.
pb.tick(3, True)
img = pb.screen.image
if img is not None:
    img.resize((img.width * 3, img.height * 3)).save(SHOT)

st = mem.read_overworld_state()
bt = mem.read_battle_state()
NAMES = {51: "Forest", 13: "Route2", 2: "Pewter", 14: "Route3", 50: "ForestGate", 1: "ViridianCity"}
hp = pb.memory[0xD16C] << 8 | pb.memory[0xD16D]
maxhp = pb.memory[0xD18D] << 8 | pb.memory[0xD18E]
lvl = pb.memory[0xD18C]
party_n = pb.memory[0xD163]
print(f"map={st.map_id}({NAMES.get(st.map_id,'?')}) pos=({st.x},{st.y}) "
      f"battle_type={getattr(bt,'battle_type',0)} textbox={getattr(st,'text_box_active',False)} "
      f"leadL={lvl} hp={hp}/{maxhp} party={party_n}")
pb.stop()
