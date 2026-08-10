#!/usr/bin/env python3
"""Fused triton FP8 blockwise dequant kernel: correctness + speed vs torch."""
import torch, os, time
import triton, triton.language as tl

@triton.jit
def _dequant_fp8_block_kernel(
    w_ptr, sc_ptr, out_ptr,
    N, K, KNB,            # KNB = K//128 (num K-blocks)
    BLOCK_N: tl.constexpr,
):
    e = tl.program_id(0)
    pk = tl.program_id(1)        # one program per (expert, K-block)
    n = tl.arange(0, BLOCK_N)
    k = pk * 128 + tl.arange(0, 128)
    nk = n[:, None] * K + k[None, :]
    base = e * N * K
    w = tl.load(w_ptr + base + nk)                       # fp8 [BLOCK_N,128]
    # scale: per 128x128 block -> here 1 K-block, N split into N/128 groups
    blk_n = n // 128
    sbase = e * (N // 128) * KNB + blk_n * KNB + pk
    s = tl.load(sc_ptr + sbase)                          # [BLOCK_N]
    out = w.to(tl.float32) * s[:, None]
    tl.store(out_ptr + base + nk, out.to(tl.bfloat16))


def dequant_fp8_block_triton(w, sc, BLOCK_N=1024, num_warps=8):
    e, N, K = w.shape
    KNB = K // 128
    assert N <= BLOCK_N or N == BLOCK_N
    out = torch.empty((e, N, K), dtype=torch.bfloat16, device=w.device)
    grid = (e, K // 128)
    _dequant_fp8_block_kernel[grid](w, sc, out, N, K, KNB,
                                    BLOCK_N=N, num_warps=num_warps)
    return out


def main():
    torch.manual_seed(0)
    N, K = 1024, 2048
    B = 128
    nb, kb = N // B, K // B
    dev = "cuda"
    w_bf = torch.randn(8, N, K, dtype=torch.bfloat16, device=dev) * 0.3
    # blockwise max scale [8, nb, kb]
    sc = w_bf.view(8, nb, B, kb, B).float().abs().amax(dim=(2, 4)).clamp(min=1e-6)
    sc = sc * torch.arange(1, kb + 1, device=dev).float().view(1, 1, kb)
    w_fp8 = (w_bf.view(8, nb, B, kb, B).float() / sc.view(8, nb, 1, kb, 1)).clamp(-1, 1).to(torch.float8_e4m3fn).view(8, N, K)

    # reference dequant (torch)
    ref = (w_fp8.to(torch.bfloat16).view(8, nb, B, kb, B) * sc.to(torch.bfloat16).view(8, nb, 1, kb, 1)).view(8, N, K)
    # triton
    tri = dequant_fp8_block_triton(w_fp8, sc)
    err = (tri.float() - ref.float()).abs().mean().item() / (ref.float().abs().mean().item() + 1e-9)
    print(f"correctness triton vs torch dequant: rel_err = {err:.5f}  {'<<< OK' if err < 0.02 else 'MISMATCH'}")

    # speed
    def torch_deq():
        return (w_fp8.to(torch.bfloat16).view(8, nb, B, kb, B) * sc.to(torch.bfloat16).view(8, nb, 1, kb, 1)).view(8, N, K)
    def t(name, f, reps=300):
        for _ in range(20): f()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): f()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        print(f"  {name:24s}: {(t1-t0)*1e6/reps:.0f} us")
    t("torch dequant", torch_deq)
    t("triton dequant", lambda: dequant_fp8_block_triton(w_fp8, sc))

if __name__ == "__main__":
    main()
