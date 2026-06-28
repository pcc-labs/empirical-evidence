import json

from autotune.genome import base_genome
from autotune.nudge_sft import (
    Winner,
    build_dataset,
    build_pair_example,
    split_train_valid,
    write_sft_data,
)
from autotune.nudge_steer import parse_genome_response
from tests.conftest import make_verdict


def _winner(*, reward=6, score=100.0, on_story=True, stuck=8):
    v = make_verdict(furthest_beat=6, story_reward=reward, on_story=on_story, score=score)
    object.__setattr__(v, "fitness", {"final_map_id": 1, "stuck_count": stuck, "turns": 400})
    return Winner(params={**base_genome(), "stuck_threshold": stuck}, verdict=v)


def test_build_pair_example_user_prompt_matches_inference(story):
    # The training user-turn must be the same prompt shape the loop uses at inference.
    source = _winner(reward=5, score=10.0, stuck=3)
    ex = build_pair_example(source, {**base_genome(), "stuck_threshold": 5}, story)
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert "propose one modified genome" in ex["messages"][1]["content"].lower()
    assert "Viridian City" in ex["messages"][1]["content"]


def test_build_pair_example_target_is_flat_and_parseable(story):
    # Guards the silent-fallback bug: a wrapped {"genome": ...} answer parses to None.
    ex = build_pair_example(_winner(stuck=3), {**base_genome(), "stuck_threshold": 5}, story)
    parsed = parse_genome_response(ex["messages"][-1]["content"])
    assert parsed is not None
    assert parsed["stuck_threshold"] == 5


def test_build_dataset_pairs_weaker_to_best(story):
    winners = [
        _winner(reward=6, score=90.0, stuck=15),  # best
        _winner(reward=6, score=50.0, stuck=10),  # same beat, weaker score
        _winner(reward=5, score=10.0, stuck=3),  # earlier beat
    ]
    examples = build_dataset(winners, story)
    assert len(examples) == 2  # both weaker winners pair toward the best
    for ex in examples:
        assert parse_genome_response(ex["messages"][-1]["content"])["stuck_threshold"] == 15


def test_build_dataset_skips_off_story(story):
    winners = [
        _winner(reward=6, score=90.0, stuck=15),
        _winner(reward=5, score=10.0, stuck=3),
        _winner(reward=5, score=99.0, on_story=False, stuck=4),  # off-story: excluded
    ]
    examples = build_dataset(winners, story)
    assert len(examples) == 1


def test_build_dataset_empty_without_gradient(story):
    # Two on-story winners tied at the same rank -> no weaker source -> no pairs.
    winners = [_winner(reward=6, score=50.0, stuck=5), _winner(reward=6, score=50.0, stuck=6)]
    assert build_dataset(winners, story) == []
    assert build_dataset([_winner()], story) == []  # fewer than two -> empty


def _pair_winners():
    return [
        _winner(reward=6, score=90.0, stuck=15),
        _winner(reward=5, score=10.0, stuck=3),
    ]


def test_split_train_valid_deterministic():
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(10)]
    a_train, a_valid = split_train_valid(rows, valid_frac=0.2, seed=1)
    b_train, b_valid = split_train_valid(rows, valid_frac=0.2, seed=1)
    assert a_train == b_train and a_valid == b_valid
    assert len(a_valid) == 2 and len(a_train) == 8


def test_split_train_valid_tiny_input():
    train, valid = split_train_valid([{"x": 1}], seed=1)
    assert valid == []
    assert len(train) == 1


def test_write_sft_data_creates_files(tmp_path, story):
    examples = build_dataset(_pair_winners(), story)
    assert examples  # sanity: the pair winners produce at least one example
    train_path, valid_path = write_sft_data(tmp_path / "sft", examples, valid_frac=0.5, seed=1)
    assert train_path.exists() and valid_path.exists()
    train_lines = train_path.read_text().strip().splitlines()
    assert all(json.loads(line)["messages"] for line in train_lines)


def test_write_sft_data_mirrors_when_too_few(tmp_path, story):
    # A single example -> valid mirrors train so MLX-LM still finds valid.jsonl.
    train_path, valid_path = write_sft_data(tmp_path / "sft", build_dataset(_pair_winners(), story))
    assert valid_path.read_text().strip()  # non-empty
