from pathlib import Path

from autotune.tetris_corpus import read_run

FIXTURE = Path(__file__).parent / "fixtures" / "tetris_run" / "20260904-000000-abcdef"


def test_read_run_yields_one_record_per_graded_event_with_a_board():
    records, exclusions = read_run(FIXTURE)
    assert [r.turn for r in records] == [1, 2, 3]
    assert exclusions == {"late": 1}


def test_read_run_fills_state_choice_and_outcome():
    records, _ = read_run(FIXTURE)
    r = records[1]
    assert r.run_id == "20260904-000000-abcdef"
    assert (r.arm, r.model, r.harness, r.effort, r.seed, r.mode) == (
        "pi/gemma4:26b/features/off+fixed", "pi/gemma4:26b", "features", "off", 100, "paused"
    )
    assert (r.piece, r.next_piece, r.chosen, r.reason) == ("O", "T", [0, 0], "Left corner.")
    assert len(r.board) == 18 and r.board[17] == "...####..."
    assert r.grade["rank"] == 2 and r.grade["genome"]["w_lines"] == 0.760666
    assert r.outcome == {
        "final_score": 120, "lines": 3, "pieces_placed": 4, "topped_out": True, "pieces_after": 2,
    }


def test_read_run_skips_graded_events_without_a_board(tmp_path):
    import json
    import shutil

    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    lines = (run / "events.jsonl").read_text().splitlines()
    out = []
    for line in lines:
        e = json.loads(line)
        if e["event_type"] == "placement_graded" and e["turn"] == 2:
            del e["data"]["board"]
        out.append(json.dumps(e))
    (run / "events.jsonl").write_text("\n".join(out) + "\n")
    records, exclusions = read_run(run)
    assert [r.turn for r in records] == [1, 3]
    assert exclusions == {"late": 1, "no_board": 1}


def test_read_run_parses_a_live_session_without_arm_fields(tmp_path):
    import json
    import shutil

    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    lines = (run / "events.jsonl").read_text().splitlines()
    session = json.loads(lines[0])
    session["data"] = {"phase": "start", "policy": "pi/gemma4:26b/features/off", "mode": "live", "level": 0, "timer_div": 7}
    lines[0] = json.dumps(session)
    (run / "events.jsonl").write_text("\n".join(lines) + "\n")
    records, _ = read_run(run)
    r = records[0]
    assert (r.model, r.harness, r.effort, r.seed, r.mode) == ("pi/gemma4:26b", "features", "off", 7, "live")
    assert r.arm == "pi/gemma4:26b/features/off+fixed"  # from meta.json when the session has none
