import json

from autotune.forest_story import ForestSignals, ForestVerdict, score_forest
from autotune.genome import base_genome
from autotune.nudge_sft import (
    ForestWinner,
    Winner,
    assemble_forest_corpus,
    build_dataset,
    build_forest_dataset,
    build_forest_mutation_prompt,
    build_forest_pair_example,
    build_pair_example,
    split_train_valid,
    write_corpus,
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


def test_map_pair_example_tagged_nav_by_default(story):
    source = _winner(reward=5, score=10.0, stuck=3)
    ex = build_pair_example(source, {**base_genome(), "stuck_threshold": 5}, story)
    assert ex["domains"] == ["nav"]
    assert set(ex) == {"messages", "domains"}


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


# --- forest-keyed SFT: pair weaker -> strongest forest crossing from the same start -----------
# The follower drives nav; the genome varies survival, so forest crossings differ by how far they
# survive. These improvement pairs teach "emit the genome that survived furthest" — the same
# rejection-sampling shape as the map-story builder, keyed on the forest reward instead.


def _forest_winner(*, reward, trainer_wins=0, turns=400, crossed=False, run_thr=0.2):
    v = ForestVerdict(
        furthest_beat=1,
        furthest_beat_name="Enter Viridian Forest",
        beats_passed=int(reward),
        per_beat=(1, 0, 0, 0, 0, 0, 0, 0),
        reward=float(reward),
        crossed=crossed,
        signals=ForestSignals(
            entered_forest=reward >= 1,
            trainer_wins=trainer_wins,
            max_bag_count=0,
            sign_read=False,
            exited=crossed,
        ),
    )
    return ForestWinner(
        params={**base_genome(), "hp_run_threshold": run_thr},
        verdict=v,
        fitness={"turns": turns},
    )


def test_forest_pair_example_target_is_flat_and_parseable():
    src = _forest_winner(reward=2, run_thr=0.2)
    target = _forest_winner(reward=5, trainer_wins=2, crossed=True, run_thr=0.45)
    ex = build_forest_pair_example(src, target)
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    parsed = parse_genome_response(ex["messages"][-1]["content"])
    assert parsed is not None and parsed["hp_run_threshold"] == 0.45


def test_forest_mutation_prompt_describes_forest_and_survival():
    prompt = build_forest_mutation_prompt(base_genome(), _forest_winner(reward=2).verdict)
    low = prompt.lower()
    assert "forest" in low
    assert "hp_run_threshold" in low  # the survival lever must be surfaced to the model


def test_build_forest_dataset_pairs_weaker_to_best():
    winners = [
        _forest_winner(reward=5, trainer_wins=2, crossed=True, run_thr=0.4),  # best: crossed
        _forest_winner(reward=3, trainer_wins=1, run_thr=0.3),
        _forest_winner(reward=2, trainer_wins=0, run_thr=0.2),
    ]
    examples = build_forest_dataset(winners)
    assert len(examples) == 2  # both weaker crossings pair toward the best
    for ex in examples:
        assert parse_genome_response(ex["messages"][-1]["content"])["hp_run_threshold"] == 0.4


def test_build_forest_dataset_breaks_reward_ties_on_survival():
    # Same reward, but one beat more catchers -> it is the stronger target.
    winners = [
        _forest_winner(reward=3, trainer_wins=2, run_thr=0.45),  # best by trainer_wins tiebreak
        _forest_winner(reward=3, trainer_wins=1, run_thr=0.25),
    ]
    examples = build_forest_dataset(winners)
    assert len(examples) == 1
    assert parse_genome_response(examples[0]["messages"][-1]["content"])["hp_run_threshold"] == 0.45


def test_build_forest_dataset_skips_runs_that_never_entered():
    winners = [
        _forest_winner(reward=4, trainer_wins=2, crossed=True, run_thr=0.4),
        _forest_winner(reward=0, run_thr=0.1),  # never entered the forest -> not comparable
    ]
    assert build_forest_dataset(winners) == []  # only one in-forest run -> no gradient


def test_build_forest_dataset_empty_without_gradient():
    winners = [
        _forest_winner(reward=3, trainer_wins=1, turns=400, run_thr=0.2),
        _forest_winner(reward=3, trainer_wins=1, turns=400, run_thr=0.3),
    ]
    assert build_forest_dataset(winners) == []  # identical rank -> no weaker source


def test_assemble_forest_corpus_pairs_within_state():
    by_state = {
        "lv13": [
            _forest_winner(reward=4, trainer_wins=2, crossed=True, run_thr=0.4),
            _forest_winner(reward=2, run_thr=0.2),
        ],
        "lv9": [
            _forest_winner(reward=3, trainer_wins=1, run_thr=0.35),
            _forest_winner(reward=1, run_thr=0.15),
        ],
    }
    examples = assemble_forest_corpus(by_state)
    assert len(examples) == 2  # one pair per state, concatenated


# --- domain tagging (issue #10) -----------------------------------------------------------------


def _fw(params, events, turns=500):
    return ForestWinner(params=params, verdict=score_forest(events), fitness={"turns": turns})


def _enter():
    return {"event_type": "overworld", "turn": 0, "data": {"map_id": 51}}


def _twin():
    return {"event_type": "battle_outcome", "turn": 1, "data": {"battle_type": 2, "won": True}}


def _exit():
    return {"event_type": "overworld", "turn": 2, "data": {"map_id": 13}}


def test_forest_pair_example_carries_pair_domains():
    weak = _fw({"hp_run_threshold": 0.6}, [_enter()])
    strong = _fw({"hp_run_threshold": 0.1}, [_enter(), _twin(), _exit()])
    ex = build_forest_pair_example(weak, strong)
    assert ex["domains"] == ["nav", "battle"]
    assert set(ex) == {"messages", "domains"}


def test_forest_dataset_examples_are_tagged():
    weak = _fw({"hp_run_threshold": 0.6}, [_enter()])
    strong = _fw({"hp_run_threshold": 0.1}, [_enter(), _twin(), _exit()])
    examples = build_forest_dataset([weak, strong])
    assert examples and all(isinstance(e["domains"], list) and e["domains"] for e in examples)


def test_write_corpus_keeps_domains(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "x"}], "domains": ["battle"]}]
    path = write_corpus(tmp_path / "corpus.jsonl", rows)
    on_disk = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert on_disk == rows


def test_write_sft_data_strips_domains(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": str(i)}], "domains": ["nav"]}
        for i in range(5)
    ]
    train_path, valid_path = write_sft_data(tmp_path, rows)
    for p in (train_path, valid_path):
        for ln in p.read_text().splitlines():
            assert set(json.loads(ln)) == {"messages"}
