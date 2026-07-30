"""DL: M=1-optimized FP8 GEMV kernel for decode projections.

Memory-bound GEMV (one warp per output N, coalesced weight-row read).
Targets ~2µs/GEMV vs dlblas GEMM's ~600µs (M=1 tile waste).
"""
from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, is_arch_support_pdl, load_jit


@cache_once
def _jit_fp8_gemv_module() -> object:
    return load_jit(
        "fp8_gemv",
        is_arch_support_pdl(),
        cuda_files=["elementwise/fp8_gemv.cuh"],
        cuda_wrappers=[("fp8_gemv", "fp8_gemv<true>")],
    )


def fp8_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 GEMV for M=1 decode: out[N] = scale[N] * dot(x[K], weight[N,K]_fp8).

    Parameters
    ----------
    x      : [K] bf16, the single-token activation (M=1).
    weight : [N, K] fp8_e4m3, per-channel quantized weight (row-major).
    scale  : [N] fp32, per-channel scale.

    Returns
    -------
    out    : [N] bf16.
    """
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert weight.dtype == torch.float8_e4m3fn, f"weight must be fp8_e4m3, got {weight.dtype}"
    assert scale.dtype == torch.float32, f"scale must be fp32, got {scale.dtype}"
    K = x.shape[-1]
    N = weight.shape[0]
    assert weight.shape[1] == K, f"weight {weight.shape} vs x K={K}"
    assert scale.shape[0] == N

    out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    module = _jit_fp8_gemv_module()
    module.fp8_gemv(out, x.contiguous(), weight.contiguous(), scale.contiguous())
    return out
