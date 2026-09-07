"""The gate classes are measured sentences; the newest carries both sides of the Saffron guard."""

from autotune.handoff_corpus import GATES, classify_gate


def test_thirsty_guard_matches_both_sentences_the_window_shows():
    # the refused step's window, then the guard spoken to, then the same guard once cleared
    assert classify_gate("I on guard duty. Gee, I thi")[0] == "thirsty_guard"
    assert classify_gate("I thirsty! I want something to drink!")[0] == "thirsty_guard"
    assert classify_gate("Hi, thanks for the cool drinks!") is None


def test_every_gate_class_has_a_measured_clear_and_a_unique_name():
    names = [cls for cls, _pat, _clears in GATES]
    assert len(names) == len(set(names))
    assert all(len(clears) > 40 for _cls, _pat, clears in GATES)
