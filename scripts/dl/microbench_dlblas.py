#!/usr/bin/env python3
"""Microbench gptq_dlblas_gemmex: is 270us/call pure dispatch, or GPU sync?
Decides whether batching MoE GEMMs will help."""
import os, time, torch
import torch.ops

# load _dl_C
for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p):
        torch.ops.load_library(p); break
assert hasattr(torch.ops, "_dl_C"), "_dl_C not loaded"

dev = "cuda"
# real Qwen3.5-35B MoE w13 shape: weight [1024, 2048] FP8, scales [1024, 16]
K, N = 2048, 1024   # in=hidden=2048, out=2*moe_inter=1024
M = 1               # decode (1 token)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
# FP8 weights as float8_e4m3fn (same byte layout as model checkpoint)
w_bytes = torch.randint(0, 255, (N, K), dtype=torch.uint8, device=dev)
w = w_bytes.view(torch.float8_e4m3fn)                                   # [1024, 2048]
ws = torch.ones(N // 128, K // 128, dtype=torch.float32, device=dev)     # [8, 16] blockwise
wt = w.t().contiguous()  # [2048, 1024] — what the model passes (w.t())

def gemm():
    return torch.ops._dl_C.gptq_dlblas_gemmex(x, wt, ws, ws, quant_type=2, bit=8)

# warmup
for _ in range(10): gemm()
torch.cuda.synchronize()

# 1) GPU latency of ONE gemm (CUDA event, isolates GPU time from dispatch)
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record(); gemm(); e1.record(); torch.cuda.synchronize()
print(f"[1] single GEMM GPU time (CUDA event): {e0.elapsed_time(e1)*1000:.1f} us")

# 2) dispatch rate in a tight loop, NO sync between (measures host dispatch cost)
N_LAUNCH = 1000
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_LAUNCH): gemm()
t1 = time.perf_counter()
print(f"[2] {N_LAUNCH} back-to-back dispatches: {(t1-t0)*1e6/N_LAUNCH:.1f} us/call (host), {(t1-t0)*1000:.1f} ms total")

# 3) same but with sync after EACH (measures GPU latency bound)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):  # fewer, sync is expensive
    gemm(); torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"[3] 100 dispatches w/ sync each: {(t1-t0)*1e6/100:.1f} us/call (GPU-bound)")

# 4) Compare: a plain aten GEMM dispatch rate (baseline dispatch cost)
wbf = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_LAUNCH): torch.mm(x, wbf.t())
t1 = time.perf_counter()
print(f"[4] {N_LAUNCH} aten::mm dispatches (baseline): {(t1-t0)*1e6/N_LAUNCH:.1f} us/call (host)")
torch.cuda.synchronize()

# 5) how many gemms can overlap before CPU blocks? launch burst then sync once
BURST = 760  # like one decode step
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(BURST): gemm()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"[5] {BURST} GEMMs burst then 1 sync: {(t1-t0)*1000:.1f} ms total = {(t1-t0)*1e6/BURST:.1f} us effective")
