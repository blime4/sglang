#!/usr/bin/env python3
"""DL: V4 FP4 MoE cuDNN microbench — isolate H1(cuDNN compute slow) vs H2(host
descriptor overhead) and sweep GEMM tiles, WITHOUT loading the 149GB model.

Loads layer-0 routed-expert weights from the checkpoint (real FP4 data, block=32
MXFP4: E2M1 packed int8 + e8m0 per-32 scale), assembles [E,N,Kpacked], and calls
torch.ops.sgl_kernel.invoke_fused_moe_opt directly in a loop.

Decomposition per call:
  host_us  = perf_counter around the call (NO cuda sync)  -> CPU enqueue + cuDNN
             descriptor create/set/query/workspace-alloc/destroy cost (H2)
  gpu_us   = CUDA events around the call                    -> real kernel time (H1)
If host_us >> gpu_us  -> H2 (descriptor overhead dominates)  -> cache descriptor
If gpu_us  >> host_us -> H1 (cuDNN GEMM genuinely slow)      -> write fused kernel

Run (single free card):
  source /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sdk-0401/env.sh
  CUDA_VISIBLE_DEVICES=8 python scripts/dl/v4_moe_cudnn_microbench.py
"""
import os, time, json, struct, sys

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
LAYER = int(os.environ.get("V4_MB_LAYER", "0"))
# Full (no-TP) routed-expert count on one card for the microbench. Real per-rank
# under TP8 shards N (inter); the op's host overhead is N/E-shape-insensitive and
# the GEMM-tile sweep is still validly comparable. Default 256 = full expert set.
E = int(os.environ.get("V4_MB_EXPERTS", "256"))
ITERS = int(os.environ.get("V4_MB_ITERS", "50"))


def _load_expert_tensors(layer, e_count):
    """Return assembled FP4 weight + e8m0 scale tensors for one MoE layer
    (real checkpoint data, via safetensors.safe_open)."""
    from safetensors import safe_open
    import torch

    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def _stack(proj):
        keys = [f"layers.{layer}.ffn.experts.{i}.{proj}.weight" for i in range(e_count)]
        skeys = [f"layers.{layer}.ffn.experts.{i}.{proj}.scale" for i in range(e_count)]
        ws, ss = [None] * e_count, [None] * e_count
        # group by shard, load all needed keys per open
        shards_w, shards_s = {}, {}
        for k in keys:
            shards_w.setdefault(wm[k], []).append(k)
        for sk in skeys:
            shards_s.setdefault(wm[sk], []).append(sk)
        for sh, ks in shards_w.items():
            with safe_open(os.path.join(MODEL, sh), framework="pt", device="cpu") as f:
                for k in ks:
                    i = int(k.split(".experts.")[1].split(".")[0])
                    ws[i] = f.get_tensor(k)
        for sh, ks in shards_s.items():
            with safe_open(os.path.join(MODEL, sh), framework="pt", device="cpu") as f:
                for k in ks:
                    i = int(k.split(".experts.")[1].split(".")[0])
                    t = f.get_tensor(k)
                    # cuDNN needs e8m0; carry raw bits as uint8 through stack/cat,
                    # reinterpret to float8_e8m0fnu at the end (_scale_view).
                    if t.dtype not in (torch.uint8, torch.int8):
                        t = t.view(torch.uint8)
                    ss[i] = t
        wt = torch.stack(ws, dim=0).contiguous()
        st = torch.stack(ss, dim=0).contiguous()
        return wt, st

    w1, s1 = _stack("w1")   # [E, 2048, 2048] int8  (out=2048, in_packed=2048 -> logical in=4096)
    w3, s3 = _stack("w3")   # same as w1
    w2, s2 = _stack("w2")   # [E, 4096, 1024] int8  (out=4096, in_packed=1024 -> logical in=2048)
    # w13 = gate_up: concat along out-dim -> [E, 2*2048=4096, 2048]
    w13 = torch.cat([w1, w3], dim=1).contiguous()
    s13 = torch.cat([s1, s3], dim=1).contiguous()
    return w13, s13, w2, s2


def main():
    import torch
    import sgl_kernel  # noqa: F401 — registers torch.ops.sgl_kernel.invoke_fused_moe_opt
    torch.zeros(1, device="cuda")  # init ctx
    dev = torch.device("cuda")
    print(f"[mb] model={MODEL} layer={LAYER} E={E} iters={ITERS}", flush=True)
    w13, s13, w2, s2 = _load_expert_tensors(LAYER, E)
    # Scale stays uint8 (kByte -> CUDNN_DATA_UINT8): this cuDNN build has no
    # CUDNN_DATA_FP8_E8M0, so the op carries e8m0 bytes as uint8 and interprets
    # them via quant_type=FP4_W4A8. Weights are int8 (packed FP4 nibbles).
    s13v, s2v = s13, s2  # already uint8 from _load_expert_tensors
    # Optional TP8 per-rank shaping: runtime shards the inter (N) dim.
    # w13 [E, 2*inter, Kpacked] -> take 1/TP of each gate/up half; w2 [E, hidden, inter_packed]
    # -> take 1/TP of the inter_packed dim. Only GEMM *size* must match runtime for
    # timing (weight values irrelevant to kernel time).
    TP = int(os.environ.get("V4_MB_TP", "1"))
    if TP > 1:
        half = w13.shape[1] // 2
        per = half // TP
        w13 = torch.cat([w13[:, 0:per], w13[:, half:half + per]], dim=1).contiguous()
        s13 = torch.cat([s13[:, 0:per], s13[:, half:half + per]], dim=1).contiguous()
        w2 = w2[:, :, : w2.shape[2] // TP].contiguous()
        s2 = s2[:, :, : s2.shape[2] // TP].contiguous()
        s13v, s2v = s13, s2
        print(f"[mb] TP{TP} per-rank slice applied", flush=True)
    w13 = w13.to(dev); s13v = s13v.to(dev); w2 = w2.to(dev); s2v = s2v.to(dev)
    print(f"[mb] w13={tuple(w13.shape)} {w13.dtype}  s13={tuple(s13v.shape)} {s13v.dtype}", flush=True)
    print(f"[mb] w2 ={tuple(w2.shape)} {w2.dtype}  s2 ={tuple(s2v.shape)} {s2v.dtype}", flush=True)

    # decode M=1, topk=6 routed experts. hidden (x dim) = w13 logical K = 4096.
    topk = int(os.environ.get("V4_MB_TOPK", "6"))
    hidden = 4096
    inter = w13.shape[1] // 2          # 2048 (full, no TP)
    M = 1
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=dev)
    topk_w = torch.rand(M, topk, dtype=torch.float32, device=dev)
    # trivial dispatch (use_moe_cu) backed by a large zero buffer (OOB-safe).
    PAD = 4096
    _srt = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    _eid = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    _npp = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    # mxfp4 quant tuple: (fp8_w8a8, int8_w8a16, int4_w4a16, mxfp4_w4a16)
    QF = (False, False, False, True)

    G = torch.ops.sgl_kernel.invoke_fused_moe_opt
    torch.cuda.synchronize()

    def run_once(BM, BN, BK, block_shape):
        c13 = torch.empty(M, topk, 2 * inter, dtype=torch.bfloat16, device=dev)
        G(x, w13, c13, None, s13v, None, topk_w, topk_ids,
          _srt, _eid, _npp, False, topk, BM, BN, BK,
          QF[0], QF[1], QF[2], QF[3], block_shape, M)
        he = torch.nn.functional.silu(c13[:, :, :inter]) * c13[:, :, inter:]
        _M2 = M * topk
        c2 = torch.empty(_M2, 1, hidden, dtype=torch.bfloat16, device=dev)
        G(he.reshape(_M2, inter), w2, c2, None, s2v, None,
          topk_w.reshape(-1, 1), topk_ids.reshape(-1, 1).to(torch.int32),
          _srt, _eid, _npp, True, 1, BM, BN, BK,
          QF[0], QF[1], QF[2], QF[3], block_shape, _M2)
        return c2

    # ---- warmup (first call compiles/JITs dleol cuDNN graph) ----
    try:
        run_once(16, 128, 128, [128, 128])
        torch.cuda.synchronize()
        print("[mb] warmup OK (16/128/128, block [128,128])", flush=True)
    except Exception as e:
        print(f"[mb] warmup (16/128/128,[128,128]) FAILED: {e!r}", flush=True)

    # ---- bench harness: host (no-sync) vs gpu (events) over ITERS ----
    def bench(BM, BN, BK, block_shape):
        # host-only timing (no sync) — captures CPU enqueue + descriptor churn
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            run_once(BM, BN, BK, block_shape)
        t1 = time.perf_counter()
        host_total_us = (t1 - t0) * 1e6
        # gpu timing — events around the loop (true GPU time)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            starts[i].record()
            run_once(BM, BN, BK, block_shape)
            ends[i].record()
        torch.cuda.synchronize()
        gpu_us = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1e3 / ITERS
        return host_total_us / ITERS, gpu_us

    print(f"\n{'config':<28}{'host_us/call':>14}{'gpu_us/call':>14}{'verdict':>16}", flush=True)
    print("-" * 72, flush=True)
    configs = [
        # GEMM tile (BM,BN,BK) ; quant block_shape
        ((16, 128, 128), [128, 128], "sglang-default"),
        ((32, 64, 32), [128, 128], "vllm-decode"),
        ((64, 64, 32), [128, 128], "vllm-prefill"),
        ((128, 128, 128), [128, 128], "big-tile"),
        ((16, 128, 128), [1, 32], "sglang blk32"),
    ]
    for tile, blk, label in configs:
        BM, BN, BK = tile
        try:
            h, g = bench(BM, BN, BK, blk)
            verdict = "H2 host-bound" if h > g else "H1 gpu-bound"
            print(f"{label+' '+str(tile):<28}{h:>14.1f}{g:>14.1f}{verdict:>16}", flush=True)
        except Exception as e:
            print(f"{label+' '+str(tile):<28}FAILED: {e!r}", flush=True)

    # ---- E-scaling probe: does gpu_us grow with expert count? ----
    # If gpu_us ~ E  -> cuDNN scans ALL experts (routing inefficiency) -> a custom
    # kernel touching only the 6 routed experts wins big. If E-independent -> cuDNN
    # already routes efficiently -> limited headroom.
    if TP > 1:
        print(f"\n--- E-scaling (TP{TP}, tile 16/128/128) ---", flush=True)
        print(f"{'E':<8}{'gpu_us/call':>14}{'bytes_6exp(MB)':>18}{'eff_GBps(6exp)':>16}", flush=True)
        w13_full, s13_full, w2_full, s2_full = w13, s13v, w2, s2v  # TP-sliced versions
        for E_test in (8, 32, 128, 256):
            try:
                _w13 = w13_full[:E_test]; _s13 = s13_full[:E_test]
                _w2 = w2_full[:E_test]; _s2 = s2_full[:E_test]
                _tids = torch.randint(0, E_test, (M, topk), dtype=torch.int32, device=dev)
                # patch closure locals by re-binding via a run fn
                def run_e():
                    c13 = torch.empty(M, topk, 2 * inter, dtype=torch.bfloat16, device=dev)
                    G(x, _w13, c13, None, _s13, None, topk_w, _tids,
                      _srt, _eid, _npp, False, topk, 16, 128, 128,
                      QF[0], QF[1], QF[2], QF[3], [128, 128], M)
                    he = torch.nn.functional.silu(c13[:, :, :inter]) * c13[:, :, inter:]
                    _M2 = M * topk
                    c2 = torch.empty(_M2, 1, hidden, dtype=torch.bfloat16, device=dev)
                    G(he.reshape(_M2, inter), _w2, c2, None, _s2, None,
                      topk_w.reshape(-1, 1), _tids.reshape(-1, 1).to(torch.int32),
                      _srt, _eid, _npp, True, 1, 16, 128, 128,
                      QF[0], QF[1], QF[2], QF[3], [128, 128], _M2)
                for _ in range(3):
                    run_e()
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
                for i in range(ITERS):
                    starts[i].record(); run_e(); ends[i].record()
                torch.cuda.synchronize()
                g = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1e3 / ITERS
                # bytes for the 6 routed experts (w13+w2), FP4 = 0.5 byte/elem
                b13 = 6 * inter * 2 * hidden * 0.5
                b2 = 6 * hidden * inter * 0.5
                mb = (b13 + b2) / 1e6
                gbps = mb / (g / 1e3) / 1e3
                print(f"{E_test:<8}{g:>14.1f}{mb:>18.2f}{gbps:>16.1f}", flush=True)
            except Exception as e:
                print(f"{E_test:<8}FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
