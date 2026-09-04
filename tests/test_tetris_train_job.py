import json

from autotune.tetris_train_job import (
    generate_tier1,
    grade_answers,
    parse_placement,
    to_prompt_completion,
    upload_adapter,
    upload_tier1,
)


def test_to_prompt_completion_splits_system_and_user_from_the_assistant_turn():
    """TRL's conversational prompt/completion format masks the prompt tokens from
    the loss -- without this split SFTTrainer computes loss over the whole
    sequence (the fixed system prompt and the 18-row board), not the ~25-token
    placement, which with 366 rows may be the difference between learning
    something and nothing."""
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": '{"rotation": 0, "col": 3, "reason": "x"}'},
        ]
    }
    assert to_prompt_completion(row) == {
        "prompt": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        "completion": [
            {"role": "assistant", "content": '{"rotation": 0, "col": 3, "reason": "x"}'},
        ],
    }


def test_to_prompt_completion_does_not_mutate_the_on_disk_row():
    """The corpus builder's on-disk format is not this job's concern -- the
    conversion must be read-only over the loaded row."""
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ]
    }
    before = json.dumps(row)
    to_prompt_completion(row)
    assert json.dumps(row) == before


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


class _FakeApi:
    def __init__(self):
        self.calls: list[tuple] = []

    def create_repo(self, *a, **kw):
        self.calls.append(("create_repo", a, kw))

    def upload_folder(self, *a, **kw):
        self.calls.append(("upload_folder", a, kw))

    def create_tag(self, *a, **kw):
        self.calls.append(("create_tag", a, kw))


def test_upload_adapter_pushes_the_adapter_folder_before_anything_tier1(tmp_path):
    """The adapter must be pushed as its own commit right after training -- a
    tier-1 crash after this point must never cost the weights."""
    api = _FakeApi()
    upload_adapter(api, tmp_path, "bdougie/x", "corpus-1")
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder"]
    _, args, kw = api.calls[1]
    assert kw["repo_id"] == "bdougie/x" and kw["path_in_repo"] == "adapter"
    assert "corpus-1" in kw["commit_message"]
    # No tag yet -- that only happens once tier-1 has also been pushed.
    assert not (tmp_path / "eval_tier1.json").exists()


def test_generate_tier1_grades_only_what_was_generated_before_a_mid_loop_crash():
    """generate() failing partway through -- OOM, a chat-template edge case --
    must not discard the answers already produced; it must not propagate either,
    since the weights this grades are already safely pushed by then."""
    rows = _rows()
    calls = []

    def flaky_generate(row):
        calls.append(row)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return '{"rotation": 0, "col": 3}'

    tier1, answers = generate_tier1(rows, flaky_generate)
    assert len(calls) == 3  # stopped at the crash; did not skip ahead or retry
    assert answers == ['{"rotation": 0, "col": 3}', '{"rotation": 0, "col": 3}']
    assert tier1["n"] == 2 and tier1["parse_failures"] == 0


def test_generate_tier1_grades_every_row_when_nothing_crashes():
    rows = _rows()
    tier1, answers = generate_tier1(rows, lambda row: '{"rotation": 0, "col": 3}')
    assert len(answers) == len(rows) == tier1["n"]


def test_upload_tier1_writes_the_json_then_pushes_a_second_commit_and_tags(tmp_path):
    api = _FakeApi()
    tier1 = {"n": 2, "top1": 0.5}
    upload_tier1(api, tmp_path, "bdougie/x", "corpus-1", tier1)
    assert json.loads((tmp_path / "eval_tier1.json").read_text()) == tier1
    names = [c[0] for c in api.calls]
    assert names == ["upload_folder", "create_tag"]
    _, args, kw = api.calls[0]
    assert kw["repo_id"] == "bdougie/x" and kw["path_in_repo"] == "adapter"
    assert "tier-1" in kw["commit_message"] or "corpus-1" in kw["commit_message"]
    _, args, kw = api.calls[1]
    assert kw.get("tag") == "corpus-1" or args[1:2] == ("corpus-1",)


def test_resolve_corpus_prefers_the_corpus_dir_under_a_local_parent(tmp_path):
    """CORPUS_DIR may be the parent tetris_corpus --out / hf download wrote the
    corpus under, or the <corpus_id>/ directory itself. Either way the Hub is
    never touched."""
    from autotune.tetris_train_job import resolve_corpus

    corpus = tmp_path / "corpus-1"
    corpus.mkdir()
    (corpus / "train.jsonl").write_text("{}\n")

    def never(_):
        raise AssertionError("downloaded")

    assert resolve_corpus("corpus-1", str(tmp_path), never) == corpus
    assert resolve_corpus("corpus-1", str(corpus), never) == corpus


def test_resolve_corpus_fails_loudly_on_an_empty_local_dir(tmp_path):
    import pytest

    from autotune.tetris_train_job import resolve_corpus

    with pytest.raises(FileNotFoundError, match="CORPUS_DIR"):
        resolve_corpus("corpus-1", str(tmp_path), lambda _: tmp_path)


def test_resolve_corpus_downloads_when_no_local_dir_is_given(tmp_path):
    from autotune.tetris_train_job import resolve_corpus

    seen = []

    def download(cid):
        seen.append(cid)
        return tmp_path

    assert resolve_corpus("corpus-1", None, download) == tmp_path / "corpus-1"
    assert seen == ["corpus-1"]


def test_should_upload_is_off_for_a_local_corpus_unless_asked():
    """A local run must not need HF_TOKEN; UPLOAD=1 opts back in, UPLOAD=0 opts
    the HF Jobs path out."""
    from autotune.tetris_train_job import should_upload

    assert should_upload({}) is True
    assert should_upload({"CORPUS_DIR": "data/tetris"}) is False
    assert should_upload({"CORPUS_DIR": "data/tetris", "UPLOAD": "1"}) is True
    assert should_upload({"UPLOAD": "0"}) is False


def test_generate_tier1_carries_the_crash_so_a_partial_result_is_not_a_bad_student():
    """`{"n": 0, "top1": 0.0}` after a crash must not read as a student that
    scored zero: the error and the intended row count travel with the numbers."""
    rows = _rows()

    def crash(row):
        raise RuntimeError("boom")

    tier1, answers = generate_tier1(rows, crash)
    assert answers == [] and tier1["n"] == 0
    assert "boom" in tier1["generation_error"] and tier1["eval_rows"] == len(rows)

    tier1, _ = generate_tier1(rows, lambda row: '{"rotation": 0, "col": 3}')
    assert tier1["generation_error"] is None and tier1["eval_rows"] == tier1["n"]
