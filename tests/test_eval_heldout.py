import json

from autotune.eval_heldout import parse_json_answer, score_rows


def test_parse_json_answer_extracts_first_object():
    assert parse_json_answer('noise {"win": true} trailing') == {"win": True}
    assert parse_json_answer("no json here") is None


def _row(domain, user, answer):
    return {
        "domain": domain,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def test_score_rows_accuracy_per_domain():
    rows = [
        _row("battle-outcome", "u1", json.dumps({"win": True, "recommendation": "fight"})),
        _row("battle-outcome", "u2", json.dumps({"win": False, "recommendation": "flee"})),
        _row("move-choice", "u3", json.dumps({"move": "Ember"})),
        _row("narrator", "u4", "not gated"),
    ]

    # fake model always answers win=True / move=Ember
    def predict(system, user):
        return '{"win": true, "recommendation": "fight", "move": "Ember"}'

    scores = score_rows(rows, predict)
    assert scores == {"battle-outcome": 0.5, "move-choice": 1.0}
