"""Fast CUDA sanity check for the RTX 5090 (Blackwell sm_120) before training.

Run after ``uv sync`` to confirm torch sees the GPU and a bf16 matmul actually launches on
sm_120 (a torch wheel without cu128 kernels will import fine but fail at the first kernel).

    uv run python smoke_cuda.py
"""

import sys
import time

import torch

print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible — check `nvidia-smi` and the torch cu128 install")

d = torch.device("cuda:0")
print("device:", torch.cuda.get_device_name(0))
print("cap:", torch.cuda.get_device_capability(0), "(expect (12, 0) for Blackwell)")
print("vram total:", torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")

# bf16 matmul — the dtype the LoRA SFT path trains in.
try:
    t0 = time.perf_counter()
    a = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"bf16 4096^2 matmul ok  dt={(time.perf_counter() - t0) * 1000:.1f} ms  "
          f"sum={c.float().sum().item():.3e}")
except Exception as e:  # noqa: BLE001
    sys.exit(f"bf16 matmul FAILED — {type(e).__name__}: {e}")

print("CUDA smoke passed.")
