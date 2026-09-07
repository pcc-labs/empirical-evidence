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


def test_parse_json_answer_keeps_nested_values():
    # the Forger's answer carries a list; the flat scan must not truncate it
    text = 'sure {"body": "npc", "items": ["POTION"], "gate": null, "outcome": "handed"} end'
    assert parse_json_answer(text) == {
        "body": "npc", "items": ["POTION"], "gate": None, "outcome": "handed"}
    assert parse_json_answer('{ not json } {"gate": "surf_no_landing"}') == {
        "gate": "surf_no_landing"}
    assert parse_json_answer("[1, 2]") is None


def test_score_rows_scores_forger_fields_separately():
    def npc(body, outcome):
        return json.dumps({"body": body, "outcome": outcome, "items": [], "gate": None})

    rows = [
        _row("npc-dialogue", "u1", npc("npc", "talk")),
        _row("npc-dialogue", "u2", npc("trainer", "fought-won")),
        _row("npc-dialogue", "u3", npc("npc", "stale")),
        _row("gate-text", "u4", json.dumps({"gate": "surf_no_landing", "clears_with": "x"})),
        _row("gate-text", "u5", json.dumps({"gate": "strength_boulder", "clears_with": "y"})),
        _row("narrator", "u6", "not gated"),
    ]

    # fake model: always body=npc outcome=talk; always gate=surf_no_landing
    def predict(system, user):
        if user in ("u4", "u5"):
            return json.dumps({"gate": "surf_no_landing", "clears_with": "?"})
        return npc("npc", "talk")

    scores = score_rows(rows, predict)
    assert scores == {
        "npc-dialogue/body": 2 / 3,
        "npc-dialogue/outcome": 1 / 3,
        "gate-text/gate": 0.5,
    }


def test_score_rows_unparseable_answer_misses_every_field():
    rows = [_row("npc-dialogue", "u", json.dumps(
        {"body": "npc", "outcome": "talk", "items": [], "gate": None}))]
    scores = score_rows(rows, lambda s, u: "no json")
    assert scores == {"npc-dialogue/body": 0.0, "npc-dialogue/outcome": 0.0}


def test_majority_baseline_is_the_mode_of_the_field():
    from autotune.eval_heldout import majority_baseline, majority_baselines, row_counts

    def npc(outcome):
        return json.dumps({"body": "npc", "outcome": outcome, "items": [], "gate": None})

    rows = [
        _row("npc-dialogue", "u1", npc("talk")),
        _row("npc-dialogue", "u2", npc("talk")),
        _row("npc-dialogue", "u3", npc("stale")),
        _row("npc-dialogue", "u4", npc("fought-won")),
        _row("gate-text", "u5", json.dumps({"gate": "g", "clears_with": "c"})),
        _row("battle-outcome", "u6", json.dumps({"win": True})),
        _row("narrator", "u7", "not gated"),
    ]
    assert majority_baseline(rows, "npc-dialogue", "outcome") == {
        "label": "talk", "accuracy": 0.5, "n": 4}
    assert majority_baseline(rows, "gate-text", "gate") == {"label": "g", "accuracy": 1.0, "n": 1}
    assert majority_baseline(rows, "handoff", "x") is None
    assert majority_baselines(rows) == {
        "npc-dialogue/outcome": {"label": "talk", "accuracy": 0.5, "n": 4}}
    assert row_counts(rows) == {"npc-dialogue": 4, "gate-text": 1, "battle-outcome": 1}


def test_gate_passed_requires_beating_base_and_majority():
    from autotune.eval_heldout import gate_passed

    base = {"battle-outcome": 0.6, "npc-dialogue/body": 0.5, "npc-dialogue/outcome": 0.4}
    majority = {"npc-dialogue/outcome": {"label": "talk", "accuracy": 0.56, "n": 84}}

    tuned = {"battle-outcome": 0.6, "npc-dialogue/body": 0.9, "npc-dialogue/outcome": 0.7}
    assert gate_passed(base, tuned, majority)

    # beats base but not the always-"talk" baseline
    tuned = {"battle-outcome": 0.6, "npc-dialogue/body": 0.9, "npc-dialogue/outcome": 0.56}
    assert not gate_passed(base, tuned, majority)

    # beats the baseline but regresses a battle domain
    tuned = {"battle-outcome": 0.5, "npc-dialogue/body": 0.9, "npc-dialogue/outcome": 0.7}
    assert not gate_passed(base, tuned, majority)

    # no baseline rows at all: base-vs-tuned is the whole gate
    assert gate_passed({"battle-outcome": 0.6}, {"battle-outcome": 0.6}, {})
