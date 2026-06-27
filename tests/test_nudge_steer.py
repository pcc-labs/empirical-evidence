from autotune.genome import base_genome
from autotune.nudge_steer import (
    build_mutation_prompt,
    genome_diffs,
    parse_genome_response,
    propose_next_genome,
    render_genome_block,
    write_genome_block,
    write_nudge_note,
)
from tests.conftest import make_verdict


def test_genome_diffs_only_changes():
    g = base_genome()
    g["stuck_threshold"] = 12
    assert genome_diffs(g) == {"stuck_threshold": 12}


def test_build_mutation_prompt_mentions_target(story):
    v = make_verdict(furthest_beat=4, story_reward=4)
    history = [{"generation": 0, "reward": 4}]
    prompt = build_mutation_prompt(base_genome(), v, story, history=history)
    assert "Viridian City" in prompt
    assert "stuck_threshold" in prompt


def test_parse_genome_response_plain_json():
    out = parse_genome_response('{"stuck_threshold": 12, "door_cooldown": 6}')
    assert out["stuck_threshold"] == 12
    assert out["door_cooldown"] == 6


def test_parse_genome_response_code_fenced():
    text = "```json\n{\"stuck_threshold\": 99}\n```"
    assert parse_genome_response(text)["stuck_threshold"] == 20  # clamped


def test_parse_genome_response_garbage():
    assert parse_genome_response("no json here") is None
    assert parse_genome_response(None) is None
    assert parse_genome_response("{not valid}") is None


def test_parse_genome_response_ignores_unknown_keys():
    assert parse_genome_response('{"bogus": 1}') is None


def test_heuristic_exploit_on_success(story):
    v = make_verdict(reached_target=True)
    winner = base_genome()
    nxt = propose_next_genome(winner, v, story, seed=1)
    assert nxt == winner  # unchanged on success


def test_heuristic_perturbs_when_blocked_deterministically(story):
    v = make_verdict(reached_target=False)
    winner = base_genome()
    a = propose_next_genome(winner, v, story, seed=3)
    b = propose_next_genome(winner, v, story, seed=3)
    assert a == b  # deterministic given seed
    assert a != winner  # something changed


def test_proposer_is_used_when_given(story):
    v = make_verdict(reached_target=False)

    def fake_proposer(_prompt: str) -> str:
        return '{"stuck_threshold": 5}'

    nxt = propose_next_genome(base_genome(), v, story, proposer=fake_proposer, seed=0)
    assert nxt["stuck_threshold"] == 5


def test_proposer_failure_falls_back_to_heuristic(story):
    v = make_verdict(reached_target=True)
    nxt = propose_next_genome(base_genome(), v, story, proposer=lambda _p: "garbage", seed=0)
    assert nxt == base_genome()  # heuristic exploit


def test_write_nudge_note_appends(tmp_path):
    v = make_verdict(furthest_beat=6, story_reward=6)
    notes = tmp_path / "notes.md"
    g = base_genome()
    g["door_cooldown"] = 4
    line = write_nudge_note(notes, v, g, stamp="2026-06-26")
    assert "door_cooldown" in line
    assert notes.read_text().strip().endswith(line)
    # Appends rather than overwrites.
    write_nudge_note(notes, v, g, stamp="2026-06-27")
    assert len(notes.read_text().strip().splitlines()) == 2


def test_render_genome_block_roundtrips():
    block = render_genome_block({"stuck_threshold": 8})
    assert "autotune:genome" in block
    assert '"stuck_threshold": 8' in block


def test_write_genome_block_creates_and_replaces(tmp_path):
    notes = tmp_path / "notes.md"
    g1 = base_genome()
    g1["door_cooldown"] = 10
    write_genome_block(notes, g1)
    assert '"door_cooldown": 10' in notes.read_text()

    # Writing again replaces the block rather than appending a second one.
    g2 = base_genome()
    g2["door_cooldown"] = 5
    write_genome_block(notes, g2)
    text = notes.read_text()
    assert text.count("autotune:genome") == 1
    assert '"door_cooldown": 5' in text
    assert '"door_cooldown": 10' not in text


def test_write_genome_block_preserves_other_notes(tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("# Agent Notes\n- keep this line\n")
    write_genome_block(notes, base_genome())
    assert "keep this line" in notes.read_text()
