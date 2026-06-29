from autotune.ollama_proposer import _extract_content, _strip_think


def test_strip_think_removes_reasoning_block():
    assert _strip_think("<think>hmm</think>{\"a\": 1}") == '{"a": 1}'
    assert _strip_think("  {\"a\": 1}  ") == '{"a": 1}'
    assert _strip_think("") == ""


def test_strip_think_multiline():
    text = "<think>\nlong\nreasoning\n</think>\n{\"x\": 2}"
    assert _strip_think(text) == '{"x": 2}'


def test_extract_content_pulls_message_and_strips_think():
    resp = {"message": {"role": "assistant", "content": "<think>plan</think>{\"speed\": 3}"}}
    assert _extract_content(resp) == '{"speed": 3}'


def test_extract_content_handles_missing_fields():
    assert _extract_content({}) == ""
    assert _extract_content({"message": None}) == ""
    assert _extract_content({"message": {}}) == ""
