"""Phase B capture tool: drive the agent Pewter->gym->Brock and save a battle-start pre_brock
state (optionally at a poked level with potions). See docs/brock-phase-b-findings.md.
Usage: uv run python scripts/capture_pre_brock.py [LEVEL] [OUT_NAME]  (LEVEL 0 = organic)
"""
"""Dead-simple gym solver from the CLEAN state: only ever climb UP (col-4 path is clear), fight
battles as they come, talk Brock at the top. NEVER press down/exit (that's what dumped us to map 0).
Capture pre_brock.state at Brock's battle start (enemy_level>=12)."""
import os, json
os.environ["EVOLVE_PARAMS"]=json.dumps({"hp_run_threshold":0.35,"hp_heal_threshold":0.4,"unknown_move_score":10.0,"status_move_score":0.0})
from autotune.forest_follow import _import_pk
from autotune.game_profile import detect_profile
from autotune.party import OFF_CUR_HP, OFF_MAX_HP
PokemonAgent,WorldMap=_import_pk()
ROM=os.environ.get("ROM_PATH") or "../pokemon-kafka/rom/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
STATES="autotune/states"
import sys
LEVEL=int(sys.argv[1]) if len(sys.argv)>1 else 0        # 0 = organic (no poke)
OUT=sys.argv[2] if len(sys.argv)>2 else "pre_brock.state"
ag=PokemonAgent(ROM,strategy="medium")
with open(f"{STATES}/gym_clean.state","rb") as f: ag.pyboy.load_state(f)
ag.pyboy.tick(2,False)
if LEVEL>0:
    from autotune.party import set_lead_level, set_bag
    set_lead_level(ag.pyboy, LEVEL)          # poke level + recompute stats + full heal
    set_bag(ag.pyboy, [(0x14, 40)])          # 40 Potions so the agent can heal through Brock
    print(f"poked lead -> L{LEVEL} + 40 potions")
prof=detect_profile(ag.pyboy); LEAD=prof.party_base
mx=ag.pyboy.memory[LEAD+OFF_MAX_HP]*256+ag.pyboy.memory[LEAD+OFF_MAX_HP+1]; ag.pyboy.memory[LEAD+OFF_CUR_HP]=mx>>8; ag.pyboy.memory[LEAD+OFF_CUR_HP+1]=mx&0xFF
def s(): return ag.memory.read_overworld_state()
def pos(): st=s(); return st.map_id,st.x,st.y
def bt(): return ag.memory.read_battle_state()
def step(d):
    m,x,y=pos(); ag.controller.move(d); m2,x2,y2=pos()
    if (m2,x2,y2)==(m,x,y): ag.controller.move(d); m2,x2,y2=pos()
    return (m2,x2,y2)!=(m,x,y)
NB={"up":(3,4),"left":(4,3),"right":(4,5)}
def walkable(d):
    ag.collision_map.update(ag.pyboy); r,c=NB[d]; return ag.collision_map.grid[r][c]==1

def heal_full():
    # keep the lead alive through the Camper attrition (traversal aid only — the captured
    # pre_brock state is still an honest full-HP L13 battle start): top up party + battle HP.
    ag.pyboy.memory[LEAD+OFF_CUR_HP]=mx>>8; ag.pyboy.memory[LEAD+OFF_CUR_HP+1]=mx&0xFF
    bm=ag.pyboy.memory[prof.addr_battle_max_hp_hi]*256+ag.pyboy.memory[prof.addr_battle_max_hp_hi+1]
    if bm>0: ag.pyboy.memory[prof.addr_battle_hp_hi]=bm>>8; ag.pyboy.memory[prof.addr_battle_hp_hi+1]=bm&0xFF

captured=False; prev=None; nomove=0
for n in range(4000):
    heal_full()
    b=bt()
    if b.battle_type==2 and b.enemy_level>=12:
        import os as _os; _os.makedirs(f"{STATES}/brock", exist_ok=True)
        outp = f"{STATES}/brock/{OUT}" if LEVEL>0 else f"{STATES}/pre_brock.state"
        with open(outp,"wb") as f: ag.pyboy.save_state(f)
        print(f"*** BROCK: enemy_lvl={b.enemy_level} hp={b.enemy_hp} -> {outp} ***"); captured=True; break
    if b.battle_type!=0:
        ag.run_battle_turn(); continue
    m,x,y=pos()
    if m!=54: print(f"left gym to {m} at ({x},{y})"); break
    if y<=2:
        # at the top row — Brock is right here; talk up to challenge
        ag.controller.press("a"); ag.controller.wait(5)
        if bt().battle_type!=0: continue
    if step("up"):
        pass
    else:
        ag.controller.press("a"); ag.controller.wait(4)   # maybe Brock/trainer above → talk
        if bt().battle_type!=0: continue
        # up blocked & no battle: sidestep (right/left ONLY — never down/out) to find the lane
        moved=False
        for d in ("right","left"):
            if walkable(d) and step(d): moved=True; break
        if not moved: pass
    cur=pos(); nomove=0 if cur!=prev else nomove+1; prev=cur
    if n%40==0: print(f"step {n}: pos={cur}")
    if nomove>150: print(f"stuck at {cur}"); break
print(f"RESULT: pos={pos()} captured={captured}")
ag.pyboy.stop()
