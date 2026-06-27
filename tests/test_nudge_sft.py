import json

from autotune.genome import base_genome
from autotune.nudge_sft import (
    Winner,
    build_dataset,
    build_example,
    split_train_valid,
    summarize_fitness,
    write_sft_data,
)
from tests.conftest import make_verdict


def _winner(reward=6, on_story=True):
    v = make_verdict(furthest_beat=6, story_reward=reward, on_story=on_story)
    object.__setattr__(v, "fitness", {"final_map_id": 1, "stuck_count": 3, "turns": 400})
    return Winner(params=base_genome(), verdict=v)


def test_summarize_fitness_subset():
    out = summarize_fitness({"final_map_id": 1, "turns": 5, "ignored": 9})
    assert out == {"final_map_id": 1, "turns": 5}


def test_build_example_is_chat_with_genome(story):
    ex = build_example(_winner(), story)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    answer = json.loads(ex["messages"][-1]["content"])
    assert answer["genome"]["stuck_threshold"] == base_genome()["stuck_threshold"]
    assert "Viridian City" in ex["messages"][1]["content"]


def test_build_dataset_skips_off_story(story):
    winners = [_winner(on_story=True), _winner(on_story=False)]
    examples = build_dataset(winners, story)
    assert len(examples) == 1


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
    examples = build_dataset([_winner(), _winner()], story)
    train_path, valid_path = write_sft_data(tmp_path / "sft", examples, valid_frac=0.5, seed=1)
    assert train_path.exists() and valid_path.exists()
    train_lines = train_path.read_text().strip().splitlines()
    assert all(json.loads(line)["messages"] for line in train_lines)


def test_write_sft_data_mirrors_when_too_few(tmp_path, story):
    # A single example -> valid mirrors train so MLX-LM still finds valid.jsonl.
    train_path, valid_path = write_sft_data(tmp_path / "sft", build_dataset([_winner()], story))
    assert valid_path.read_text().strip()  # non-empty
