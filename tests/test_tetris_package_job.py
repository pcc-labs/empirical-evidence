from autotune.tetris_package_job import deploy_instruction


def test_deploy_instruction_names_deploy_gguf_sh_not_ollama_pull():
    """`ollama pull hf.co/<repo>:Q4_K_M` selects by quant tag alone, so from the
    second corpus onward it resolves *some* Q4_K_M file in the adapter repo, not
    necessarily this one -- and `ollama cp` labels it as the requested corpus
    regardless. The printed deploy step must name the unambiguous script."""
    msg = deploy_instruction("bdougie/gemma-4-E4B-tetris-lora", "20260904-abc123def456")
    assert "deploy_gguf.sh 20260904-abc123def456" in msg
    assert "ollama pull" not in msg
