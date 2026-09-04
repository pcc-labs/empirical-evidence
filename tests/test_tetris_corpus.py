import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from autotune.tetris_corpus import (
    EVAL_SEEDS,
    LIVE_DEADLINE_S,
    TRAIN_SEED_MAX,
    TRAIN_SEED_MIN,
    apply_filters,
    board_array,
    build_corpus,
    corpus_id,
    eval_row,
    read_run,
    sft_row,
    split,
)

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
    session["data"] = {
        "phase": "start", "policy": "pi/gemma4:26b/features/off", "mode": "live",
        "level": 0, "timer_div": 7,
    }
    lines[0] = json.dumps(session)
    (run / "events.jsonl").write_text("\n".join(lines) + "\n")
    records, _ = read_run(run)
    r = records[0]
    assert (r.model, r.harness, r.effort, r.seed, r.mode) == (
        "pi/gemma4:26b", "features", "off", 7, "live",
    )
    assert r.arm == "pi/gemma4:26b/features/off+fixed"  # from meta.json when the session has none


def _records(n, *, run_id="r", seed=100, topped_out=False, chosen_value=-2.0, best_value=-1.0):
    base, _ = read_run(FIXTURE)
    proto = base[0]
    out = []
    for turn in range(1, n + 1):
        out.append(
            replace(
                proto,
                run_id=run_id,
                turn=turn,
                seed=seed,
                grade={**proto.grade, "chosen_value": chosen_value, "best_value": best_value},
                outcome={
                    **proto.outcome, "pieces_placed": n, "topped_out": topped_out,
                    "pieces_after": n - turn,
                },
            )
        )
    return out


def test_death_spiral_drops_exactly_the_last_five_of_a_topped_out_run():
    kept, dropped = apply_filters(_records(12, topped_out=True))
    assert [r.turn for r in kept] == list(range(1, 8))
    assert dropped == {"death_spiral": 5}


def test_a_survived_run_keeps_every_decision():
    kept, dropped = apply_filters(_records(12, topped_out=False))
    assert len(kept) == 12 and dropped == {}


def test_top_out_veto_drops_only_a_lethal_choice_with_a_survivable_alternative():
    lethal = _records(1, run_id="a", chosen_value=-1e6, best_value=-3.5)
    forced = _records(1, run_id="b", chosen_value=-1e6, best_value=-1e6)
    fine = _records(1, run_id="c", chosen_value=-2.0, best_value=-1.0)
    kept, dropped = apply_filters(lethal + forced + fine)
    assert [r.run_id for r in kept] == ["b", "c"]
    assert dropped == {"top_out_veto": 1}


def test_fixture_run_after_filters():
    records, _ = read_run(FIXTURE)
    kept, dropped = apply_filters(records)
    # 3 records, topped out: the last 5 are dropped -> nothing survives the spiral.
    assert kept == [] and dropped == {"death_spiral": 3}


def test_split_refuses_rows_on_the_wrong_side_of_the_seed_line():
    train_ok = _records(2, seed=TRAIN_SEED_MIN)
    valid_ok = _records(2, seed=EVAL_SEEDS[0])
    stray = _records(2, seed=50)
    train, valid, excluded = split(train_ok + valid_ok + stray)
    assert {r.seed for r in train} == {TRAIN_SEED_MIN}
    assert {r.seed for r in valid} == {EVAL_SEEDS[0]}
    assert excluded == {"seed_out_of_pool": 2}
    assert all(r.seed >= TRAIN_SEED_MIN for r in train)
    assert all(r.seed in EVAL_SEEDS for r in valid)


def test_split_excludes_a_wrapped_seed_above_train_seed_max():
    """PyBoy masks the seed with `& 0xFF` (pyboy/plugins/base_plugin.py), so seed
    256 is seed 0 and 257-261 replay eval seeds 1-5. A wrapped seed must be
    excluded, not filed as train just because it is numerically >= TRAIN_SEED_MIN."""
    train_ok = _records(2, seed=TRAIN_SEED_MAX)
    wrapped = _records(2, seed=TRAIN_SEED_MAX + 1)
    train, valid, excluded = split(train_ok + wrapped)
    assert {r.seed for r in train} == {TRAIN_SEED_MAX}
    assert valid == []
    assert excluded == {"seed_wraps": 2}
    assert TRAIN_SEED_MAX == 255  # seed 255 is train, seed 256 wraps to seed 0


def test_sft_user_turn_is_byte_identical_to_the_served_pi_prompt():
    from tetris_agent.pi_policy import PI_JSON_INSTRUCTIONS, PI_PROMPT_SUFFIX
    from tetris_agent.prompts import build_user_prompt, legal_placements, system_prompt_for

    r, *_ = read_run(FIXTURE)[0]
    row = sft_row(r)
    system, user, assistant = row["messages"]
    board = board_array(r.board)
    # A literal deadline, not LIVE_DEADLINE_S piped through both sides: this must
    # fail if someone edits the constant without editing what it renders to.
    expected_user = (
        build_user_prompt(
            "features", board, r.piece, r.next_piece,
            legal_placements(board, r.piece), r.turn,
            deadline_s=13.0,
        )
        + PI_PROMPT_SUFFIX
    )
    assert system == {
        "role": "system", "content": system_prompt_for("features") + PI_JSON_INSTRUCTIONS,
    }
    assert user == {"role": "user", "content": expected_user}
    assert assistant["role"] == "assistant"


def test_live_deadline_s_never_exceeds_the_served_prompts_level0_ceiling():
    """`LIVE_DEADLINE_S` must be a value the live prompt can actually produce.

    tetris's live_agent._deadline_s tops out, at level 0, at
    (ROWS - 1) * (_GRAVITY_RELOADS[0] + 1) / 60 - _EXEC_HEADROOM_S. A gravity or
    headroom change in tetris should break this build, not silently corrupt the
    corpus with a deadline line the served prompt can never say.
    """
    from tetris_agent.board import ROWS
    from tetris_agent.emulator import Emulator
    from tetris_agent.live_agent import _EXEC_HEADROOM_S

    ceiling = (ROWS - 1) * (Emulator._GRAVITY_RELOADS[0] + 1) / 60 - _EXEC_HEADROOM_S
    assert LIVE_DEADLINE_S <= ceiling


def test_sft_row_raises_on_a_harness_less_record():
    """A harness of None means a non-model arm (heuristic, random, ...); coercing
    it to "features" would silently render a real prompt for a control arm."""
    r, *_ = read_run(FIXTURE)[0]
    r = replace(r, harness=None)
    with pytest.raises(ValueError, match="harness"):
        sft_row(r)


def test_sft_assistant_turn_is_the_terse_placement_json():
    r, *_ = read_run(FIXTURE)[0]
    assistant = sft_row(r)["messages"][2]["content"]
    parsed = json.loads(assistant)
    assert parsed == {"rotation": r.chosen[0], "col": r.chosen[1], "reason": r.reason}
    assert assistant == '{"rotation": 0, "col": 3, "reason": "Flat on the floor."}'


def test_board_array_round_trips():
    r, *_ = read_run(FIXTURE)[0]
    arr = board_array(r.board)
    assert arr.shape == (18, 10) and arr.dtype == bool
    assert ["".join("#" if c else "." for c in row) for row in arr] == r.board


def test_eval_row_carries_prompt_teacher_and_the_full_ranking():
    from tetris_agent.prompts import legal_placements

    r, *_ = read_run(FIXTURE)[0]
    row = eval_row(r)
    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    assert row["messages"] == sft_row(r)["messages"][:2]
    assert row["teacher"] == r.chosen
    # One entry per legal placement, computed by the same enumeration the prompt shows.
    assert len(row["ranking"]) == len(legal_placements(board_array(r.board), r.piece))
    assert all(len(entry) == 3 for entry in row["ranking"])
    values = [v for _, _, v in row["ranking"]]
    assert values == sorted(values, reverse=True)
    assert tuple(r.chosen) in {(rot, col) for rot, col, _ in row["ranking"]}
    assert (row["turn"], row["run_id"], row["seed"]) == (r.turn, r.run_id, r.seed)


def test_eval_row_ranking_is_the_oracle_ranking():
    from tetris_agent.policy import Genome
    from tetris_agent.quality import rank_placements

    r, *_ = read_run(FIXTURE)[0]
    row = eval_row(r, ply=2)
    expected = rank_placements(board_array(r.board), r.piece, r.next_piece, Genome(), 2)
    assert [(rot, col) for rot, col, _ in row["ranking"]] == [p for p, _ in expected]


def _runs_dir(tmp_path):
    """Two copies of the fixture: one on a train seed that survives, one on eval seed 1."""
    runs = tmp_path / "runs"
    for name, seed, topped in (("train-run", 100, False), ("eval-run", 1, False)):
        dst = runs / name
        shutil.copytree(FIXTURE, dst)
        summary = json.loads((dst / "summary.json").read_text())
        summary["run_id"] = name
        summary["fitness"]["topped_out"] = topped
        (dst / "summary.json").write_text(json.dumps(summary))
        lines = (dst / "events.jsonl").read_text().splitlines()
        session = json.loads(lines[0])
        session["data"]["seed"] = seed
        lines[0] = json.dumps(session)
        (dst / "events.jsonl").write_text("\n".join(lines) + "\n")
    return runs


def test_build_corpus_writes_every_file_and_the_stats(tmp_path):
    out = build_corpus(_runs_dir(tmp_path), tmp_path / "data", ply=2)
    assert out.parent == tmp_path / "data"
    names = {p.name for p in out.iterdir()}
    assert names == {"records.jsonl", "train.jsonl", "valid.jsonl", "eval.jsonl", "stats.json"}
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(x) for x in (out / "valid.jsonl").read_text().splitlines()]
    ev = [json.loads(x) for x in (out / "eval.jsonl").read_text().splitlines()]
    # 3 records per run; the veto drops turn 3 in each -> 2 train, 2 valid.
    assert len(train) == 2 and len(valid) == 2 and len(ev) == 2
    assert all(len(row["messages"]) == 3 for row in train)
    assert all("ranking" in row for row in ev)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["runs"] == 2
    assert stats["records"] == 6
    assert stats["kept"] == 4
    assert stats["excluded"] == {"late": 2, "top_out_veto": 2}
    assert stats["per_seed"] == {"1": 2, "100": 2}
    assert stats["per_arm"] == {"pi/gemma4:26b/features/off+fixed": 6}
    assert stats["train_rows"] == 2 and stats["valid_rows"] == 2
    assert 0.0 <= stats["teacher"]["mean_regret_norm"] <= 1.0
    assert stats["teacher"]["top1_rate"] == 0.5


def _relabel_arm(run_dir, *, run_id, seed, policy, arm, model, harness, effort):
    """Point a fixture copy at a different arm/model/harness/effort, so a runs/
    directory can hold both the teacher's runs and a control arm's on one seed."""
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["run_id"] = run_id
    summary["fitness"]["topped_out"] = False
    (run_dir / "summary.json").write_text(json.dumps(summary))
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    session = json.loads(lines[0])
    session["data"] = {
        "phase": "start", "policy": policy, "arm": arm, "model": model,
        "harness": harness, "effort": effort, "mode": "paused", "seed": seed,
    }
    lines[0] = json.dumps(session)
    (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n")


def test_build_corpus_excludes_every_arm_but_the_teacher(tmp_path):
    """runs/ is the dataset of record for every arm tetris_rollout mints into --
    build_corpus must not silently fold a control arm's placements into training."""
    runs = tmp_path / "runs"
    teacher = runs / "teacher-run"
    shutil.copytree(FIXTURE, teacher)
    _relabel_arm(
        teacher, run_id="teacher-run", seed=100, policy="pi/gemma4:26b/features/off",
        arm="pi/gemma4:26b/features/off+fixed", model="pi/gemma4:26b",
        harness="features", effort="off",
    )
    control = runs / "random-run"
    shutil.copytree(FIXTURE, control)
    _relabel_arm(
        control, run_id="random-run", seed=100, policy="random", arm="random",
        model="random", harness=None, effort=None,
    )

    out = build_corpus(runs, tmp_path / "data")
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    records = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
    stats = json.loads((out / "stats.json").read_text())

    assert {r["run_id"] for r in records} == {"teacher-run"}
    assert len(train) == 2  # 3 teacher records, minus the top-out-veto drop
    assert stats["excluded"]["wrong_arm"] == 3  # every record from the random run
    assert stats["per_arm"] == {"pi/gemma4:26b/features/off+fixed": 3, "random": 3}


def test_build_corpus_selectors_can_target_a_different_arm(tmp_path):
    runs = tmp_path / "runs"
    control = runs / "random-run"
    shutil.copytree(FIXTURE, control)
    _relabel_arm(
        control, run_id="random-run", seed=100, policy="random", arm="random",
        model="random", harness="features", effort=None,
    )
    out = build_corpus(runs, tmp_path / "data", model="random", harness="features", effort=None)
    records = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
    assert {r["run_id"] for r in records} == {"random-run"}
    stats = json.loads((out / "stats.json").read_text())
    assert "wrong_arm" not in stats["excluded"]


def test_build_corpus_raises_when_it_would_mint_a_zero_row_corpus(tmp_path):
    """A zero-row corpus still wrote all five files and a corpus id, and printed
    the hf upload command, in a real 17-run runs/ that filtered to nothing. That
    is a meaningless corpus id and a job that dies on a paid GPU at
    train.jsonl[0] -- raise here instead, naming why."""
    runs = tmp_path / "runs"
    control = runs / "random-run"
    shutil.copytree(FIXTURE, control)
    _relabel_arm(
        control, run_id="random-run", seed=100, policy="random", arm="random",
        model="random", harness=None, effort=None,
    )
    with pytest.raises(RuntimeError, match="wrong_arm"):
        build_corpus(runs, tmp_path / "data")


def test_corpus_id_is_dated_and_content_addressed(tmp_path):
    records, _ = read_run(FIXTURE)
    a = corpus_id(records)
    b = corpus_id(list(reversed(records)))
    assert a == b and len(a.split("-")[1]) == 12 and a[:8].isdigit()


def test_build_corpus_skips_a_run_with_no_boards(tmp_path):
    runs = _runs_dir(tmp_path)
    old = runs / "old-run"
    shutil.copytree(FIXTURE, old)
    lines = []
    for line in (old / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        e["data"].pop("board", None)
        lines.append(json.dumps(e))
    (old / "events.jsonl").write_text("\n".join(lines) + "\n")
    out = build_corpus(runs, tmp_path / "data")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["runs"] == 3 and stats["excluded"]["no_board"] == 3
