from autotune.tetris_train_job import grade_answers, parse_placement


def test_parse_placement_reads_the_first_json_object():
    assert parse_placement('{"rotation": 2, "col": 5, "reason": "x"}') == (2, 5)
    assert parse_placement('Sure!\n{"rotation": 0, "col": 3}\ntrailing') == (0, 3)
    assert parse_placement("no json here") is None
    assert parse_placement('{"rotation": "two", "col": 5}') is None


def _rows():
    ranking = [[0, 3, -2.0], [0, 4, -2.5], [1, 0, -3.0], [1, 7, -6.0]]
    return [
        {"teacher": [0, 3], "ranking": ranking},
        {"teacher": [0, 4], "ranking": ranking},
        {"teacher": [0, 3], "ranking": ranking},
        {"teacher": [1, 7], "ranking": ranking},
    ]


def test_grade_answers_scores_rank_regret_and_agreement():
    answers = [
        '{"rotation": 0, "col": 3}',  # rank 1, regret 0, agrees with teacher
        '{"rotation": 1, "col": 0}',  # rank 3, regret (2.0-3.0... ) = 1.0/4.0 = 0.25
        "garbage",  # parse failure
        '{"rotation": 1, "col": 7}',  # rank 4, regret_norm 1.0, agrees with teacher
    ]
    out = grade_answers(_rows(), answers)
    assert out["n"] == 4
    assert out["parse_failures"] == 1
    assert out["top1"] == 0.25
    assert out["top3"] == 0.5
    assert abs(out["mean_regret_norm"] - (0.0 + 0.25 + 1.0) / 3) < 1e-9
    assert out["teacher_agreement"] == 0.5


def test_grade_answers_treats_an_illegal_placement_as_a_parse_failure():
    out = grade_answers(_rows()[:1], ['{"rotation": 3, "col": 9}'])
    assert out["parse_failures"] == 1 and out["top1"] == 0.0
