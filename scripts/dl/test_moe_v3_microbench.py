#!/usr/bin/env python3
"""Standalone MoE microbench (no engine, no flash-attn): compare
  (a) non-v3 invoke_fused_moe_opt  (sglang's current slow prefill path)
  (b) v3 via vLLM dl_invoke_moe_v3  (vLLM's fast path — reference)
  (c) v3 raw op + sglang mabs       (my manual call — does it segfault at M=512?)
Random FP8 weights => output garbage, but SPEED + NO-CRASH are valid.
Shapes from real Qwen3.6-35B-A3B-FP8 TP4 rank: w13=(256,256,2048) w2=(256,2048,128).
"""
import os, time
import torch

E, INTER, HIDDEN, TOPK = 256, 128, 2048, 8
DEV = "cuda:0"
BS = [128, 128]  # blockwise FP8 block


def make_weights():
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device=DEV) * 0.01).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, HIDDEN, INTER, device=DEV) * 0.01).to(torch.float8_e4m3fn)
    # blockwise scales [E, N/128, K/128]
    s1 = torch.randn(E, (2 * INTER) // 128, HIDDEN // 128, device=DEV, dtype=torch.float32)
    s2 = torch.randn(E, HIDDEN // 128, INTER // 128, device=DEV, dtype=torch.float32)
    return w1.contiguous(), w2.contiguous(), s1.contiguous(), s2.contiguous()


def make_input(M):
    x = (torch.randn(M, HIDDEN, device=DEV) * 0.1).to(torch.bfloat16)
    ti = torch.randint(0, E, (M, TOPK), device=DEV, dtype=torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=DEV, dtype=torch.float32), dim=-1)
    return x, ti, tw


def bench(fn, M, name, iters=20):
    x, ti, tw = make_input(M)
    for _ in range(5):  # warmup
        fn(x, ti, tw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, ti, tw)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"  [{name}] M={M}: {dt * 1000:.2f} ms/call  ({M / dt:.0f} tok/s)", flush=True)
    return dt


def main():
    import vllm, os
    import sgl_kernel  # registers sgl_kernel.invoke_fused_moe_opt (non-v3)
    so = os.path.join(os.path.dirname(vllm.__file__), "_dl_C.cpython-312-x86_64-linux-gnu.so")
    torch.ops.load_library(so)
    w1, w2, s1, s2 = make_weights()
    _V3 = torch.ops._dl_C.invoke_fused_moe_opt_v3
    _NV3 = torch.ops.sgl_kernel.invoke_fused_moe_opt
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size as sg_mabs,
    )
    from sglang.jit_kernel.activation import silu_and_mul

    # (b) vLLM reference
    from vllm.plugins.dl_platform_plugin.ops.dl_fused_moe import dl_invoke_moe_v3

    def fn_vllm(x, ti, tw):
        return dl_invoke_moe_v3(x, w1, w2, tw, ti, w1_scale=s1, w2_scale=s2, weight_bits=8)

    # vLLM's standalone mabs (import-safe: no flash_attn in chain)
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as vllm_mabs,
    )

    def _v3_call(A, B, C, B_scale, tw, ti, srt, eid, npp, mul_rw, top_k, BM, M):
        _V3(A.view(-1, A.shape[-1]), B, C.view(-1, top_k, C.shape[-1]), None, B_scale, None,
            tw.view(-1, top_k), ti.view(-1, top_k), srt, eid, npp, mul_rw, top_k,
            BM, 128, 128, 8, BS, M)

    # (c1) v3 raw + sglang mabs (BM=64) — reproduces engine segfault?
    def fn_v3_sg_mabs(x, ti, tw):
        M = x.shape[0]
        c13 = torch.empty(M, TOPK, 2 * INTER, dtype=x.dtype, device=x.device)
        srt, eid, npp = sg_mabs(ti, 64, E)
        _v3_call(x, w1, c13, s1, tw, ti, srt, eid, npp, False, TOPK, 64, M)
        he = silu_and_mul(c13.reshape(-1, 2 * INTER)).reshape(M, TOPK, INTER)
        c2 = torch.empty(M, TOPK, HIDDEN, dtype=x.dtype, device=x.device)
        _v3_call(he, w2, c2, s2, tw, ti, srt, eid, npp, True, 1, 64, M)
        return c2.view(M, TOPK, HIDDEN).sum(dim=1)

    # (c2) v3 raw + vLLM mabs (BM=64) — the hypothesized fix
    def fn_v3_vllm_mabs(x, ti, tw):
        M = x.shape[0]
        c13 = torch.empty(M, TOPK, 2 * INTER, dtype=x.dtype, device=x.device)
        srt, eid, npp = vllm_mabs(ti, 64, E, None)
        _v3_call(x, w1, c13, s1, tw, ti, srt, eid, npp, False, TOPK, 64, M)
        he = silu_and_mul(c13.reshape(-1, 2 * INTER)).reshape(M, TOPK, INTER)
        c2 = torch.empty(M, TOPK, HIDDEN, dtype=x.dtype, device=x.device)
        _v3_call(he, w2, c2, s2, tw, ti, srt, eid, npp, True, 1, 64, M)
        return c2.view(M, TOPK, HIDDEN).sum(dim=1)

    for M in [512, 2048]:
        print(f"\n=== M={M} ({M} tokens, E={E}, inter={INTER}, hidden={HIDDEN}) ===", flush=True)
        bench(fn_vllm, M, "vLLM dl_invoke_moe_v3 (reference)")
        # BM sweep for v3 raw + vLLM mabs (find optimal tiling)
        for bm in [16, 32, 64, 128]:
            def fn(x, ti, tw, _bm=bm):
                _M = x.shape[0]
                c13 = torch.empty(_M, TOPK, 2 * INTER, dtype=x.dtype, device=x.device)
                srt, eid, npp = vllm_mabs(ti, _bm, E, None)
                _v3_call(x, w1, c13, s1, tw, ti, srt, eid, npp, False, TOPK, _bm, _M)
                he = silu_and_mul(c13.reshape(-1, 2 * INTER)).reshape(_M, TOPK, INTER)
                c2 = torch.empty(_M, TOPK, HIDDEN, dtype=x.dtype, device=x.device)
                _v3_call(he, w2, c2, s2, tw, ti, srt, eid, npp, True, 1, _bm, _M)
                return c2.view(_M, TOPK, HIDDEN).sum(dim=1)
            try:
                bench(fn, M, f"v3 + vLLMmabs BM={bm:<3}")
            except Exception as e:
                print(f"  [v3 BM={bm}] EXC: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
