from autotune.genome import PARAM_BOUNDS, base_genome
from autotune.harvest import build_genome_population, resolve_states
from autotune.nudge_sft import Winner, assemble_corpus
from autotune.nudge_steer import parse_genome_response
from autotune.scenario import NAV_PARAM_KEYS
from tests.conftest import make_verdict


def test_population_deterministic_and_includes_default():
    a = build_genome_population(seed=7, n=12)
    b = build_genome_population(seed=7, n=12)
    assert a == b  # deterministic for a fixed seed
    assert len(a) == 12
    assert base_genome() in a  # default is always a reference point


def test_population_genomes_are_within_bounds():
    for g in build_genome_population(seed=1, n=20):
        for key in NAV_PARAM_KEYS:
            lo, hi, _typ = PARAM_BOUNDS[key]
            assert lo <= g[key] <= hi


def test_population_diverse_not_all_default():
    pop = build_genome_population(seed=3, n=16)
    distinct = {tuple(sorted(g.items())) for g in pop}
    assert len(distinct) > 1  # not collapsed to the default


def test_population_zero_or_negative():
    assert build_genome_population(seed=1, n=0) == []
    assert build_genome_population(seed=1, n=-5) == []


def _winner(*, reward, score, on_story=True, stuck):
    v = make_verdict(furthest_beat=6, story_reward=reward, on_story=on_story, score=score)
    object.__setattr__(v, "fitness", {"final_map_id": 1, "stuck_count": stuck, "turns": 400})
    return Winner(params={**base_genome(), "stuck_threshold": stuck}, verdict=v)


def test_assemble_corpus_pairs_within_each_state(story):
    by_state = {
        "b.state": [
            _winner(reward=6, score=90.0, stuck=15),
            _winner(reward=5, score=10.0, stuck=3),
        ],
        "a.state": [
            _winner(reward=6, score=80.0, stuck=12),
            _winner(reward=4, score=5.0, stuck=4),
        ],
    }
    examples = assemble_corpus(by_state, story)
    assert len(examples) == 2  # one improvement pair per state
    for ex in examples:
        assert parse_genome_response(ex["messages"][-1]["content"]) is not None


def test_assemble_corpus_empty_when_no_gradient(story):
    by_state = {"s.state": [_winner(reward=6, score=50.0, stuck=5)]}  # single winner -> no pair
    assert assemble_corpus(by_state, story) == []


def test_resolve_states_dir_file_and_missing(tmp_path):
    (tmp_path / "route1.state").write_bytes(b"x")
    (tmp_path / "first.state").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("ignore")
    from_dir = resolve_states(str(tmp_path))
    assert len(from_dir) == 2 and all(p.endswith(".state") for p in from_dir)
    from_file = resolve_states(str(tmp_path / "route1.state"))
    assert len(from_file) == 1 and from_file[0].endswith("route1.state")
    assert resolve_states(str(tmp_path / "nope")) == []
