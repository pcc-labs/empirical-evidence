from autotune.tetris_train_job import grade_answers, parse_placement


def test_parse_placement_reads_the_first_json_object():
    assert parse_placement('{"rotation": 2, "col": 5, "reason": "x"}') == (2, 5)
    assert parse_placement('Sure!\n{"rotation": 0, "col": 3}\ntrailing') == (0, 3)
    assert parse_placement("no json here") is None
    assert parse_placement('{"rotation": "two", "col": 5}') is None


def test_parse_placement_skips_a_stray_brace_in_a_thinking_preamble():
    """A `{` before the real object -- a thinking preamble with its own braces --
    must not make the non-greedy regex match a truncated fragment and score a
    spurious parse failure."""
    text = '{"thought": "consider the corner"} then {"rotation": 2, "col": 5}'
    assert parse_placement(text) == (2, 5)


def _rows():
    # Row 1 carries a genuinely different ranking object (different order and
    # values) from rows 0, 2, 3 -- a grader that indexed every answer against
    # rows[0]["ranking"] instead of its own row's ranking must fail this test.
    ranking_a = [[0, 3, -2.0], [0, 4, -2.5], [1, 0, -3.0], [1, 7, -6.0]]
    ranking_b = [[0, 4, -1.0], [0, 3, -2.0], [1, 7, -4.0], [1, 0, -5.0]]
    return [
        {"teacher": [0, 3], "ranking": ranking_a},
        {"teacher": [0, 4], "ranking": ranking_b},
        {"teacher": [0, 3], "ranking": ranking_a},
        {"teacher": [1, 7], "ranking": ranking_a},
    ]


def test_grade_answers_scores_rank_regret_and_agreement():
    answers = [
        '{"rotation": 0, "col": 3}',  # ranking_a index 0 -> rank 1, regret 0, agrees
        '{"rotation": 1, "col": 0}',  # ranking_b index 3 -> rank 4, regret 1.0, disagrees
        "garbage",  # parse failure
        '{"rotation": 1, "col": 7}',  # ranking_a index 3 -> rank 4, regret 1.0, agrees
    ]
    out = grade_answers(_rows(), answers)
    assert out["n"] == 4
    assert out["parse_failures"] == 1
    assert out["top1"] == 0.25  # only row 0
    assert out["top3"] == 0.25  # only row 0; rows 1 and 3 both rank 4
    assert abs(out["mean_regret_norm"] - (0.0 + 1.0 + 1.0) / 3) < 1e-9
    assert out["teacher_agreement"] == 0.5  # rows 0 and 3


def test_grade_answers_treats_an_illegal_placement_as_a_parse_failure():
    out = grade_answers(_rows()[:1], ['{"rotation": 3, "col": 9}'])
    assert out["parse_failures"] == 1 and out["top1"] == 0.0
