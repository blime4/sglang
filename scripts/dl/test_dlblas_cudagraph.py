#!/usr/bin/env python3
"""Test: is gptq_dlblas_gemmex capturable in a CUDA graph?
If yes -> enabling decode cuda-graph for 35B could give 4-8x (kills 18k dispatches).
If no  -> must fuse GEMMs instead."""
import os, torch

for p in ["../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so"]:
    if os.path.exists(p): torch.ops.load_library(p)

K, N = 2048, 1024
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
w = torch.randint(0, 255, (N, K), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).t().contiguous()
ws = torch.ones(N // 128, K // 128, dtype=torch.float32, device="cuda")
out = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")

def run():
    for _ in range(16):  # 16 dlblas GEMMs like one MoE layer
        out = torch.ops._dl_C.gptq_dlblas_gemmex(x, w, ws, ws, quant_type=2, bit=8)
    return out

# eager reference
ref = run()
torch.cuda.synchronize()

# try to capture
g = torch.cuda.CUDAGraph()
try:
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        cap = run()
    print("[capture] SUCCESS — gptq_dlblas_gemmex IS graph-capturable")
    # replay once
    g.replay(); torch.cuda.synchronize()
    # correctness: replay output == eager?
    # NOTE: with static graph, x is fixed; re-run eager with same x
    err = (cap - ref).abs().max().item()
    print(f"[correctness] max|graph - eager| = {err:.4e}  ({'OK' if err < 1e-2 else 'MISMATCH'})")
    # speed: replay vs eager
    import time
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(1000): g.replay()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    print(f"[speed] graph replay: {(t1-t0)*1e6/1000:.1f} us/replay (16 GEMMs)")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(1000): run()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    print(f"[speed] eager 16 GEMMs: {(t1-t0)*1e6/1000:.1f} us/call")
except Exception as e:
    print(f"[capture] FAILED: {type(e).__name__}: {str(e)[:200]}")
