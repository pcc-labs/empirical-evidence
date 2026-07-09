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


def test_game_label_of_reads_system_prompt():
    from autotune.eval_heldout import game_label_of

    def r(label):
        return {"messages": [{"role": "system", "content": f"advisor for a Pokemon {label} agent"}]}

    assert game_label_of(r("Yellow")) == "Yellow"
    assert game_label_of(r("Red/Blue")) == "Red/Blue"
    assert game_label_of(r("Red")) == "Red"
    assert game_label_of({"messages": [{"role": "system", "content": "no game named"}]}) == "Red"


def test_score_rows_by_game_splits_accuracy():
    from autotune.eval_heldout import score_rows_by_game

    def sysmsg(label):
        return f"You are the battle advisor for a Pokemon {label} agent."

    rows = [
        {"domain": "battle-outcome", "messages": [
            {"role": "system", "content": sysmsg("Yellow")},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": json.dumps({"win": True})}]},
        {"domain": "battle-outcome", "messages": [
            {"role": "system", "content": sysmsg("Red")},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": json.dumps({"win": False})}]},
        {"domain": "narrator", "messages": [
            {"role": "system", "content": sysmsg("Yellow")},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "ignored, not gated"}]},
    ]

    def predict(system, user):
        return json.dumps({"win": True})  # correct for Yellow row, wrong for Red row

    by_game = score_rows_by_game(rows, predict)
    assert by_game["Yellow"] == {"accuracy": 1.0, "n": 1}
    assert by_game["Red"] == {"accuracy": 0.0, "n": 1}
    assert "narrator" not in str(by_game)  # non-gated rows excluded
