"""The gate classes are measured sentences; the newest carries both sides of the Saffron guard."""

from autotune.handoff_corpus import (
    GATES,
    UNCLASSIFIED,
    UNCLASSIFIED_CAP,
    classify_gate,
    gen_gate_text,
)


def test_thirsty_guard_matches_both_sentences_the_window_shows():
    # the refused step's window, then the guard spoken to, then the same guard once cleared
    assert classify_gate("I on guard duty. Gee, I thi")[0] == "thirsty_guard"
    assert classify_gate("I thirsty! I want something to drink!")[0] == "thirsty_guard"
    assert classify_gate("Hi, thanks for the cool drinks!") is None


def test_every_gate_class_has_a_measured_clear_and_a_unique_name():
    names = [cls for cls, _pat, _clears in GATES]
    assert len(names) == len(set(names))
    assert all(len(clears) > 40 for _cls, _pat, clears in GATES)


def _refused(text, mp=70, at=(3, 3), event="supervisor.gate_text"):
    return {
        "event": event,
        "map": mp,
        "at": list(at),
        "direction": "up",
        "said": text,
        "run_id": "r",
    }


def test_a_refused_step_with_an_unknown_sentence_teaches_the_unknown_answer():
    rows = gen_gate_text([_refused("The ship set sa|The ship set sail.")])
    assert len(rows) == 1
    import json

    label = json.loads(rows[0]["messages"][-1]["content"])
    assert label["gate"] == UNCLASSIFIED and "not measured" in label["clears_with"]
    assert "'The ship set sail.'" in rows[0]["messages"][1]["content"]


def test_unknown_rows_are_capped_per_sentence_and_only_for_refused_steps():
    events = [_refused("The door is locked...", at=(i, 0)) for i in range(10)]
    events.append(_refused("999999999999999999 999999999999999999"))  # the window's digit garbage
    events.append(
        {"event": "supervisor.body_engaged", "map": 1, "at": [1, 1], "said": "I like shorts!"}
    )
    events.append(_refused("I on guard duty. Gee, I thi"))  # a known class is never capped
    rows = gen_gate_text(events)
    import json

    labels = [json.loads(r["messages"][-1]["content"])["gate"] for r in rows]
    assert labels.count(UNCLASSIFIED) == UNCLASSIFIED_CAP
    assert labels.count("thirsty_guard") == 1
    assert (
        len(rows) == UNCLASSIFIED_CAP + 1
    )  # no row for the digits, none for the body's ordinary line
