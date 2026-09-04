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

Runs as `hf jobs uv run --flavor cpu-xl autotune/tetris_package_job.py`. The
quant is Q4_K_M on purpose — the same quant the baseline `gemma4:latest` is
served at, so the tier-2 comparison measures training, not quantization.

Env: BASE_MODEL, ADAPTER_REPO, CORPUS_ID, HF_TOKEN (secret).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def sh(*cmd: str, cwd: str | None = None) -> None:
    print("[package_job] $ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=cwd, check=True)


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

    snapshot_path = snapshot_download(
        adapter_repo, revision=corpus_id, allow_patterns=["adapter/*"]
    )
    adapter = Path(snapshot_path) / "adapter"
    print(f"[package_job] merging {adapter_repo}@{corpus_id} into {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained("merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained("merged")

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

    f16 = "merged-f16.gguf"
    quant = f"gemma-4-E4B-tetris-{corpus_id}-Q4_K_M.gguf"
    sh(
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        "merged",
        "--outfile",
        f16,
        "--outtype",
        "f16",
    )
    sh("llama.cpp/build/bin/llama-quantize", f16, quant, "Q4_K_M")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=quant,
        path_in_repo=f"gguf/{quant}",
        repo_id=adapter_repo,
        commit_message=f"Q4_K_M GGUF for corpus {corpus_id}",
    )
    print(f"[package_job] uploaded {adapter_repo}/gguf/{quant}")
    print(
        f"[package_job] on the box: ollama pull hf.co/{adapter_repo}:Q4_K_M && "
        f"ollama cp hf.co/{adapter_repo}:Q4_K_M gemma4-e4b-tetris:{corpus_id}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
