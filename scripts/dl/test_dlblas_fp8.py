#!/usr/bin/env python3
# Test dlblas FP8 blockwise GEMM — the optimization kernel validation.
# Builds the C++ extension, tests correctness + speed on a decode-shape GEMM.
import os, time
import torch
from torch.utils.cpp_extension import load

SDK = os.environ["SDK_DIR"]
mod = load(
    name="dlblas_fp8_gemm",
    sources=["scripts/dl/dlblas_fp8_gemm.cu"],
    extra_include_paths=[f"{SDK}/include"],
    extra_ldflags=[f"-L{SDK}/lib", "-ldlblas", "-lcublas"],
    verbose=True,
)

M, K, N = 64, 2048, 1024
BLOCK_K, BLOCK_N = 128, 128

torch.manual_seed(0)
A_bf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
W_bf = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
ref = (A_bf.float() @ W_bf.float()).bfloat16()

# Quantize A per-token, W blockwise
A_amax = A_bf.float().abs().amax(1, keepdim=True).clamp(min=1e-4)
A_sc = (A_amax / 448.0).contiguous()
A_fp8 = (A_bf.float() / A_sc).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()

Wp = W_bf.float().reshape(K//BLOCK_K, BLOCK_K, N//BLOCK_N, BLOCK_N)
W_amax = Wp.abs().amax((1,3), keepdim=True).clamp(min=1e-4)
W_sc = (W_amax / 448.0).contiguous().view(K//BLOCK_K, N//BLOCK_N)
W_fp8 = (Wp / W_amax * 448).clamp(-448,448).to(torch.float8_e4m3fn).view(K,N).contiguous()

C = mod.dlblas_fp8_blockwise_gemm(A_fp8, W_fp8, A_sc, W_sc, BLOCK_K, BLOCK_N)
torch.cuda.synchronize()
err = (C.float() - ref.float()).abs().max().item()
rel = err / ref.float().abs().max().item()
print(f"dlblas FP8 GEMM correctness: max_err={err:.3f}, rel={rel:.4f}")
print(f"C[0,:5]={C[0,:5].tolist()}")
print(f"ref[0,:5]={ref[0,:5].tolist()}")

if rel < 0.2:
    for _ in range(5): mod.dlblas_fp8_blockwise_gemm(A_fp8, W_fp8, A_sc, W_sc, BLOCK_K, BLOCK_N)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): mod.dlblas_fp8_blockwise_gemm(A_fp8, W_fp8, A_sc, W_sc, BLOCK_K, BLOCK_N)
    torch.cuda.synchronize()
    dt = (time.time()-t0)/100
    flops = 2*M*K*N
    print(f"dlblas FP8 [{M}x{K}]x[{K}x{N}]: {dt*1000:.3f}ms, {flops/dt/1e12:.1f} TFLOP/s")
