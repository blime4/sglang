#!/usr/bin/env python3
# Validate dlblas FP8 blockwise GEMM on DLIN — the optimization kernel.
# Proves dlblasGemmExV2 (cublasGemmEx-style + dlblasExtQuantParametersV2) works + is fast.
# If fast vs sglang's triton FP8 (~2 TFLOP/s), the dlblas port is validated.
import ctypes
import os
import time

import torch

SDK = os.environ.get("SDK_DIR", "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sdk-0401")

# --- enum constants (from library_types.h / cublas_api.h) ---
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUDA_R_32F, CUDA_R_16BF = 0, 14
CUDA_R_8F_E5M2, CUDA_R_8F_E4M3 = 28, 29


class QuantParamsV2(ctypes.Structure):
    _fields_ = [
        ("a_group_size_m", ctypes.c_int32),
        ("a_group_size_k", ctypes.c_int32),
        ("a_zeropoints", ctypes.c_void_p),
        ("a_zeropoints_type", ctypes.c_int32),
        ("a_scales", ctypes.c_void_p),
        ("a_scales_type", ctypes.c_int32),
        ("b_group_size_k", ctypes.c_int32),
        ("b_group_size_n", ctypes.c_int32),
        ("b_zeropoints", ctypes.c_void_p),
        ("b_zeropoints_type", ctypes.c_int32),
        ("b_scales", ctypes.c_void_p),
        ("b_scales_type", ctypes.c_int32),
        ("c_group_size_m", ctypes.c_int32),
        ("c_group_size_n", ctypes.c_int32),
        ("c_scales", ctypes.c_void_p),
        ("c_scales_type", ctypes.c_int32),
    ]


def main():
    # torch already has a cublas handle (libdlblas = libcublas on DLIN); reuse it
    lib = ctypes.CDLL(f"{SDK}/lib/libdlblas.so", mode=ctypes.RTLD_GLOBAL)
    handle = ctypes.c_void_p(torch.cuda.current_blas_handle())

    # dlblasGemmExV2 sig: (handle, transa, transb, m, n, k, alpha*, A, Atype, lda,
    #   B, Btype, ldb, beta*, C, Ctype, ldc, computeType, algo, quantParams*)
    gemm = lib.dlblasGemmExV2
    gemm.restype = ctypes.c_int
    gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

    # Shapes: decode-step expert GEMM [M,K]x[K,N] (Qwen3.5 moe_intermediate=512 per expert, hidden=2048)
    M, K, N = 64, 2048, 1024
    BLOCK_K, BLOCK_N = 128, 128

    # Reference bf16
    torch.manual_seed(0)
    A_bf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W_bf = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    ref = (A_bf.float() @ W_bf.float()).bfloat16()  # [M,N]

    # Quantize A per-token (per-row absmax) to fp8 e4m3
    A_absmax = A_bf.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    A_scale = (A_absmax / 448.0).to(torch.float32)  # fp8 e4m3 max=448
    A_fp8 = (A_bf.float() / A_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    # Quantize W blockwise (128x128) to fp8 e4m3. W[K,N] → blocks [K/128, N/128]
    Wp = W_bf.float().reshape(K // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)
    W_absmax = Wp.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)  # [K/128,1,N/128,1]
    W_scale = (W_absmax / 448.0).to(torch.float32)
    W_fp8 = (Wp / W_scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(K, N)
    W_scale_flat = W_scale.view(K // BLOCK_K, N // BLOCK_N).contiguous()  # [K/128, N/128]

    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Build quant params. A: per-token (group_m=1). W: blockwise (group_k=128, group_n=128).
    qp = QuantParamsV2()
    qp.a_group_size_m = 1
    qp.a_group_size_k = K  # per-token (whole row = one group along K)
    qp.a_scales = ctypes.c_void_p(A_scale.data_ptr())
    qp.a_scales_type = CUDA_R_32F
    qp.b_group_size_k = BLOCK_K
    qp.b_group_size_n = BLOCK_N
    qp.b_scales = ctypes.c_void_p(W_scale_flat.data_ptr())
    qp.b_scales_type = CUDA_R_32F
    qp.c_group_size_m = 0  # no output quant
    qp.c_group_size_n = 0

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # First: validate basic bf16 GEMM works (no FP8, no quant)
    C_test = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    rc_bf = gemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
        ctypes.byref(alpha),
        ctypes.c_void_p(A_bf.data_ptr()), CUDA_R_16BF, K,
        ctypes.c_void_p(W_bf.data_ptr()), CUDA_R_16BF, N,
        ctypes.byref(beta),
        ctypes.c_void_p(C_test.data_ptr()), CUDA_R_16BF, N,
        CUDA_R_32F, 0, None,  # no quant params
    )
    torch.cuda.synchronize()
    err_bf = (C_test.float() - ref.float()).abs().max().item()
    print(f"bf16 GEMM: rc={rc_bf}, max_err={err_bf:.3f} (expect ~0)")

    # Now: FP8 blockwise GEMM
    rc = gemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        ctypes.byref(alpha),
        ctypes.c_void_p(A_fp8.data_ptr()), CUDA_R_8F_E4M3, K,
        ctypes.c_void_p(W_fp8.data_ptr()), CUDA_R_8F_E4M3, N,
        ctypes.byref(beta),
        ctypes.c_void_p(C_out.data_ptr()), CUDA_R_16BF, N,
        CUDA_R_32F, 0, ctypes.byref(qp),
    )
    torch.cuda.synchronize()
    print(f"dlblasGemmExV2 rc={rc}")

    # Correctness
    err = (C_out.float() - ref.float()).abs().max().item()
    rel = err / ref.float().abs().max().item()
    print(f"correctness: max_abs_err={err:.3f}, rel_err={rel:.4f} (expect <0.1 for FP8)")
    print(f"C_out[0,:5]={C_out[0,:5].tolist()}")
    print(f"ref[0,:5]=   {ref[0,:5].tolist()}")

    # Speed (if correct enough)
    if rel < 0.2:
        for _ in range(3):
            gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, ctypes.byref(alpha),
                 ctypes.c_void_p(A_fp8.data_ptr()), CUDA_R_8F_E4M3, K,
                 ctypes.c_void_p(W_fp8.data_ptr()), CUDA_R_8F_E4M3, K,
                 ctypes.byref(beta), ctypes.c_void_p(C_out.data_ptr()), CUDA_R_16BF, N,
                 CUDA_R_32F, 0, ctypes.byref(qp))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, ctypes.byref(alpha),
                 ctypes.c_void_p(A_fp8.data_ptr()), CUDA_R_8F_E4M3, K,
                 ctypes.c_void_p(W_fp8.data_ptr()), CUDA_R_8F_E4M3, K,
                 ctypes.byref(beta), ctypes.c_void_p(C_out.data_ptr()), CUDA_R_16BF, N,
                 CUDA_R_32F, 0, ctypes.byref(qp))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 50
        flops = 2 * M * K * N
        print(f"dlblas FP8 GEMM [{M}x{K}]x[{K}x{N}]: {dt*1000:.3f}ms, {flops/dt/1e12:.1f} TFLOP/s")


if __name__ == "__main__":
    main()
