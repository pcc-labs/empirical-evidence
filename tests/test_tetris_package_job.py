from autotune.tetris_package_job import deploy_instruction


def test_deploy_instruction_names_deploy_gguf_sh_not_ollama_pull():
    """`ollama pull hf.co/<repo>:Q4_K_M` selects by quant tag alone, so from the
    second corpus onward it resolves *some* Q4_K_M file in the adapter repo, not
    necessarily this one -- and `ollama cp` labels it as the requested corpus
    regardless. The printed deploy step must name the unambiguous script."""
    msg = deploy_instruction("bdougie/gemma-4-E4B-tetris-lora", "20260904-abc123def456")
    assert "deploy_gguf.sh 20260904-abc123def456" in msg
    assert "ollama pull" not in msg


def test_local_deploy_instruction_creates_the_ollama_tag_from_the_gguf_on_disk(tmp_path):
    """Locally the GGUF never went through the Hub, so the deploy is `ollama
    create` from a Modelfile naming the file, then the pi registration."""
    from autotune.tetris_package_job import local_deploy_instruction, ollama_tag

    quant = tmp_path / "gemma-4-E4B-tetris-c1-Q4_K_M.gguf"
    msg = local_deploy_instruction(quant, "c1")
    assert f"ollama create {ollama_tag('c1')}" in msg
    assert str(quant.resolve()) in msg
    assert "register_pi_tag.py gemma4-e4b-tetris:c1" in msg
    assert "ollama pull" not in msg


def test_quantize_binary_names_the_build_step_when_missing(tmp_path):
    import pytest

    from autotune.tetris_package_job import quantize_binary

    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        quantize_binary(str(tmp_path))
    built = tmp_path / "build" / "bin"
    built.mkdir(parents=True)
    (built / "llama-quantize").write_text("")
    assert quantize_binary(str(tmp_path)) == built / "llama-quantize"


def test_should_upload_is_off_for_a_local_adapter_unless_asked():
    from autotune.tetris_package_job import should_upload

    assert should_upload({}) is True
    assert should_upload({"ADAPTER_DIR": "out/adapter"}) is False
    assert should_upload({"ADAPTER_DIR": "out/adapter", "UPLOAD": "1"}) is True
