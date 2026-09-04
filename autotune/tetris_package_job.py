# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.11,<3",
#   "transformers>=5.12,<6",
#   "peft>=0.19,<1",
#   "huggingface_hub",
#   "gguf>=0.10",
#   "sentencepiece",
#   "protobuf",
#   "safetensors",
# ]
# ///
"""Package the tuned student for Ollama: merge the LoRA, convert to GGUF, quantize.

Runs as `hf jobs uv run --flavor cpu-xl autotune/tetris_package_job.py`, or
locally as `uv run python -m autotune.tetris_package_job` with ADAPTER_DIR and
LLAMA_CPP_DIR set. The quant is Q4_K_M on purpose — the same quant the baseline
`gemma4:latest` is served at, so the tier-2 comparison measures training, not
quantization.

Env, HF Jobs: BASE_MODEL, ADAPTER_REPO, CORPUS_ID, HF_TOKEN (secret).

Env, local: ADAPTER_DIR (the adapter tetris_train_job wrote) skips the Hub
download and, unless UPLOAD=1, the upload; LLAMA_CPP_DIR (a checkout with
build/bin/llama-quantize already built) skips apt, clone and compile; OUT_DIR is
where merged/ and the GGUFs land (default cwd). The f16 intermediate is deleted
once the quant exists — it is 16 GB per corpus.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def sh(*cmd: str, cwd: str | None = None) -> None:
    print("[package_job] $ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=cwd, check=True)


def should_upload(env) -> bool:
    """Push the GGUF to the Hub? UPLOAD=1/0 decides explicitly. Unset, a local
    adapter (ADAPTER_DIR) means a local run and nothing leaves the box."""
    explicit = env.get("UPLOAD")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes"}
    return not env.get("ADAPTER_DIR")


def quantize_binary(llama_cpp_dir: str) -> Path:
    """`llama-quantize` inside a prebuilt checkout, or a clear failure naming the
    build step -- not a FileNotFoundError from subprocess after the 16 GB merge."""
    path = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    if not path.is_file():
        raise FileNotFoundError(
            f"LLAMA_CPP_DIR={llama_cpp_dir}: no {path}; build it with "
            "`cmake -B build -DLLAMA_CURL=OFF && cmake --build build --target llama-quantize`"
        )
    return path


def gguf_name(corpus_id: str) -> str:
    return f"gemma-4-E4B-tetris-{corpus_id}-Q4_K_M.gguf"


def ollama_tag(corpus_id: str) -> str:
    return f"gemma4-e4b-tetris:{corpus_id}"


def local_deploy_instruction(quant: Path, corpus_id: str) -> str:
    """The line printed after a local package: the GGUF is already on the box, so
    the deploy is `ollama create` from a Modelfile, then the pi registration --
    what scripts/deploy_gguf.sh does after its download."""
    tag = ollama_tag(corpus_id)
    return (
        f"[package_job] on the box: printf 'FROM {quant.resolve()}\\n' > {quant.parent}/Modelfile"
        f" && ollama create {tag} -f {quant.parent}/Modelfile"
        f" && uv run python scripts/register_pi_tag.py {tag}"
    )


def deploy_instruction(adapter_repo: str, corpus_id: str) -> str:
    """The line printed after upload, naming the unambiguous deploy step.

    `ollama pull hf.co/<repo>:Q4_K_M` selects by quant tag alone: the repo holds
    one GGUF per corpus, so from the second corpus onward it resolves *some*
    Q4_K_M file, not necessarily this one, and `ollama cp` would label it as the
    requested corpus regardless. `scripts/deploy_gguf.sh` names the file by
    corpus id instead.
    """
    return (
        f"[package_job] on the box: scripts/deploy_gguf.sh {corpus_id}  # pulls the "
        f"{corpus_id} GGUF from {adapter_repo} by name, not by quant tag alone"
    )


def main() -> int:  # pragma: no cover - CPU/subprocess driver, exercised by the smoke job
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = os.environ.get("BASE_MODEL", "google/gemma-4-E4B-it")
    adapter_repo = os.environ.get(
        "ADAPTER_REPO", "bdougie/gemma-4-E4B-tetris-lora"
    )
    corpus_id = os.environ["CORPUS_ID"]
    adapter_dir = os.environ.get("ADAPTER_DIR")
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR")
    out_dir = Path(os.environ.get("OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    upload = should_upload(os.environ)

    if adapter_dir:
        adapter = Path(adapter_dir)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"ADAPTER_DIR={adapter_dir}: no adapter_config.json")
        print(f"[package_job] merging {adapter} into {base_model}")
    else:
        snapshot_path = snapshot_download(
            adapter_repo, revision=corpus_id, allow_patterns=["adapter/*"]
        )
        adapter = Path(snapshot_path) / "adapter"
        print(f"[package_job] merging {adapter_repo}@{corpus_id} into {base_model}")
    merged_dir = out_dir / "merged"
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(merged_dir))
    del merged, base

    if llama_cpp_dir:
        quantize = quantize_binary(llama_cpp_dir)
        llama_cpp = Path(llama_cpp_dir)
    else:
        sh("apt-get", "update")
        sh(
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
            "cmake",
            "build-essential",
        )
        sh(
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp",
        )
        sh(
            "cmake",
            "-B",
            "build",
            "-DGGML_NATIVE=OFF",
            "-DLLAMA_CURL=OFF",
            cwd="llama.cpp",
        )
        sh(
            "cmake",
            "--build",
            "build",
            "--target",
            "llama-quantize",
            "-j",
            cwd="llama.cpp",
        )
        llama_cpp = Path("llama.cpp")
        quantize = llama_cpp / "build" / "bin" / "llama-quantize"

    f16 = out_dir / "merged-f16.gguf"
    quant = out_dir / gguf_name(corpus_id)
    sh(
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(merged_dir),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    )
    sh(str(quantize), str(f16), str(quant), "Q4_K_M")
    if quant.is_file() and quant.stat().st_size > 0:
        f16.unlink()

    if upload:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(quant),
            path_in_repo=f"gguf/{quant.name}",
            repo_id=adapter_repo,
            commit_message=f"Q4_K_M GGUF for corpus {corpus_id}",
        )
        print(f"[package_job] uploaded {adapter_repo}/gguf/{quant.name}")
        print(deploy_instruction(adapter_repo, corpus_id))
    else:
        print(f"[package_job] wrote {quant} (no upload)")
        print(local_deploy_instruction(quant, corpus_id))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
