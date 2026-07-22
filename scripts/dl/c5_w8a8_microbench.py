#!/usr/bin/env python3
"""C-5 correctness microbench: w8a8_matmul path vs gptq_dlblas_gemmex path vs bf16 ref.

Verifiable criterion: the env-gated w8a8 path (SGLANG_DL_FP8_W8A8=1) runs without
error and its output is numerically sound (no NaN; close to the gemmex path and to
the bf16 reference, within FP8 quantization tolerance).
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from sglang.srt.layers.quantization.fp8_utils import (
    _dl_pc_cache,
    _ensure_dl_C,
    dlblas_w8a8_block_fp8_linear,
)

_ensure_dl_C()
dev = "cuda"
torch.manual_seed(0)
M, N, K, bn, bk = 8, 256, 2048, 128, 128
x = (torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.1)
w = (torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.1)
w_scale = torch.ones(N // bn, K // bk, dtype=torch.float32, device=dev)
ref = (x.float() @ w.float().t())

def run(env_w8a8):
    _dl_pc_cache.clear()
    if env_w8a8:
        os.environ["SGLANG_DL_FP8_W8A8"] = "1"
    else:
        os.environ.pop("SGLANG_DL_FP8_W8A8", None)
    os.environ.pop("SGLANG_DL_FP8_Q2", None)
    return dlblas_w8a8_block_fp8_linear(x, w, [bn, bk], w_scale).float()

def stats(a, b, name):
    nan = torch.isnan(a).any().item()
    maxd = (a - b).abs().max().item()
    rel = ((a - b).abs().mean() / (b.abs().mean() + 1e-9)).item()
    print(f"{name:18} NaN={nan}  max_abs_diff={maxd:.4f}  rel_mean_err={rel:.4f}")

g = run(False)   # gemmex (current default)
v = run(True)    # w8a8 (C-5)
stats(g, ref, "gemmex vs bf16")
stats(v, ref, "w8a8   vs bf16")
stats(v, g,   "w8a8   vs gemmex")
print("PASS" if (not torch.isnan(v).any() and (v - ref).abs().max().item() < 1.0) else "FAIL")
