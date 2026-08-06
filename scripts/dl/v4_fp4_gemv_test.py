#!/usr/bin/env python3
"""DL: correctness + perf test for the JIT fp4_grouped_gemv kernel vs cuDNN
invoke_fused_moe_opt, on REAL V4 layer-0 weights (TP8 per-rank shapes), 1 GPU.

Reference (ground truth) = fp32 dequant of FP4 (E2M1 table + e8m0 2^(raw-127),
block=32) + einsum with bf16 x. My kernel should match this closely.
cuDNN runs W4A8 (quantizes x to 8-bit) so it drifts from the true value by the
activation-quant error — included for the speed comparison and to see that drift.

Run:
  source .../sdk-0401/env.sh
  CUDA_VISIBLE_DEVICES=8 python scripts/dl/v4_fp4_gemv_test.py
"""
import os, time, json, struct, sys

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
LAYER = int(os.environ.get("V4_MB_LAYER", "0"))
TP = int(os.environ.get("V4_MB_TP", "8"))
ITERS = int(os.environ.get("V4_MB_ITERS", "100"))

DSV4_DEQUANT_FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _load_w13(layer, e_count, tp):
    from safetensors import safe_open
    import torch
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def _stack(proj):
        keys = [f"layers.{layer}.ffn.experts.{i}.{proj}.weight" for i in range(e_count)]
        skeys = [f"layers.{layer}.ffn.experts.{i}.{proj}.scale" for i in range(e_count)]
        ws, ss = [None] * e_count, [None] * e_count
        for which, keylist, out in (("w", keys, ws), ("s", skeys, ss)):
            shards = {}
            for k in keylist:
                shards.setdefault(wm[k], []).append(k)
            for sh, ks in shards.items():
                with safe_open(os.path.join(MODEL, sh), framework="pt", device="cpu") as f:
                    for k in ks:
                        i = int(k.split(".experts.")[1].split(".")[0])
                        t = f.get_tensor(k)
                        if which == "s" and t.dtype not in (torch.uint8, torch.int8):
                            t = t.view(torch.uint8)
                        out[i] = t
        return torch.stack(ws, 0).contiguous(), torch.stack(ss, 0).contiguous()

    w1, s1 = _stack("w1")
    w3, s3 = _stack("w3")
    w13 = torch.cat([w1, w3], dim=1).contiguous()
    s13 = torch.cat([s1, s3], dim=1).contiguous()
    # TP8 per-rank: shard the inter (N) dim — take 1/TP of each gate/up half.
    half = w13.shape[1] // 2
    per = half // tp
    w13 = torch.cat([w13[:, 0:per], w13[:, half:half + per]], dim=1).contiguous()
    s13 = torch.cat([s13[:, 0:per], s13[:, half:half + per]], dim=1).contiguous()
    return w13, s13


def main():
    import torch
    import sgl_kernel  # noqa: F401
    from sglang.jit_kernel.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args
    def make_name(name):
        return f"dpsk_v4_{name}"

    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    print(f"[t] model={MODEL} layer={LAYER} TP={TP}", flush=True)

    w13, s13 = _load_w13(LAYER, 256, TP)
    w13 = w13.to(dev)            # [256, 512, 2048] int8  (FP4 packed)
    s13 = s13.to(dev)            # [256, 512, 128] uint8 (e8m0)
    E, N, Kpacked = w13.shape
    K = Kpacked * 2              # 4096 logical
    print(f"[t] w13={tuple(w13.shape)} s13={tuple(s13.shape)} K={K}", flush=True)
    print(f"[t] s13 bytes: min={int(s13.min())} max={int(s13.max())} [0,0,:8]={s13[0,0,:8].tolist()}", flush=True)

    topk = 6
    topk_ids = torch.arange(topk, dtype=torch.int32, device=dev)  # experts 0..5
    x = torch.randn(K, dtype=torch.bfloat16, device=dev) * 0.1

    # ---- JIT-compile my kernel ----
    @cache_once
    def _mod():
        readonly = os.environ.get("V4_MB_READONLY", "0") == "1"
        args = make_cpp_args(is_arch_support_pdl())
        cflags = ["-use_fast_math"] + (["-DDL_READONLY_DIAG"] if readonly else [])
        return load_jit(
            make_name("fp4_grouped_gemv" + ("_ro" if readonly else "")), *args,
            cuda_files=["deepseek_v4/fp4_grouped_gemv.cuh"],
            cuda_wrappers=[("fp4_grouped_gemv", f"fp4_grouped_gemv<{args}>")],
            extra_cuda_cflags=cflags,
        )
    mod = _mod()

    out_my = torch.empty(topk, N, dtype=torch.bfloat16, device=dev)
    w13_u8 = w13.view(torch.uint8)
    mod.fp4_grouped_gemv(out_my, x, w13_u8, s13, topk_ids)
    torch.cuda.synchronize()
    print("[t] my kernel ran OK", flush=True)

    # ---- fp32 dequant reference (ground truth) ----
    table = torch.tensor(DSV4_DEQUANT_FP4_TABLE, dtype=torch.float32, device=dev)
    wi = w13_u8[topk_ids].to(torch.int32)          # [6,512,2048]
    low = wi & 0x0F
    high = (wi >> 4) & 0x0F
    vals = torch.stack([table[low], table[high]], dim=-1).reshape(topk, N, K)  # [6,512,4096]
    sc = torch.exp2(s13[topk_ids].to(torch.float32) - 127.0)                   # [6,512,128]
    sc_exp = sc.repeat_interleave(32, dim=-1)                                   # [6,512,4096]
    w_deq = (vals * sc_exp).to(torch.float32)
    out_ref = torch.einsum("k,snk->sn", x.float(), w_deq).to(torch.bfloat16)    # [6,512]
    torch.cuda.synchronize()

    # ---- cuDNN reference (W4A8) ----
    G = torch.ops.sgl_kernel.invoke_fused_moe_opt
    PAD = 4096
    _srt = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    _eid = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    _npp = torch.zeros((PAD,), dtype=torch.int32, device=dev)[:1]
    tw = torch.ones(1, topk, dtype=torch.float32, device=dev)
    c13 = torch.empty(1, topk, N, dtype=torch.bfloat16, device=dev)
    G(x.view(1, K), w13, c13, None, s13, None, tw, topk_ids.view(1, topk),
      _srt, _eid, _npp, False, topk, 16, 128, 128,
      False, False, False, True, [1, 32], 1)
    out_cudnn = c13[0]
    torch.cuda.synchronize()

    # ---- correctness ----
    def _err(a, b):
        d = (a.float() - b.float()).abs()
        rel = d / (b.float().abs() + 1e-3)
        return float(d.max()), float(rel.mean())
    my_abs, my_rel = _err(out_my, out_ref)
    cd_abs, cd_rel = _err(out_cudnn, out_ref)
    print(f"[t] my  vs ref:  max_abs={my_abs:.4f} mean_rel={my_rel:.4f}", flush=True)
    print(f"[t] cudnn vs ref: max_abs={cd_abs:.4f} mean_rel={cd_rel:.4f}", flush=True)
    print(f"[t] ref[0,:5]={out_ref[0,:5].tolist()}", flush=True)
    print(f"[t] my [0,:5]={out_my[0,:5].tolist()}", flush=True)
    print(f"[t] cd [0,:5]={out_cudnn[0,:5].tolist()}", flush=True)

    # ---- perf ----
    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            s[i].record(); fn(); e[i].record()
        torch.cuda.synchronize()
        return sum(a.elapsed_time(b) for a, b in zip(s, e)) * 1e3 / ITERS

    my_us = bench(lambda: mod.fp4_grouped_gemv(out_my, x, w13_u8, s13, topk_ids))
    def cudnn_w13():
        c = torch.empty(1, topk, N, dtype=torch.bfloat16, device=dev)
        G(x.view(1, K), w13, c, None, s13, None, tw, topk_ids.view(1, topk),
          _srt, _eid, _npp, False, topk, 16, 128, 128,
          False, False, False, True, [1, 32], 1)
    cd_us = bench(cudnn_w13)
    bytes_rw = (topk * N * Kpacked + topk * N * (K // 32) + K * 2 + topk * N * 2) / 1e6
    print(f"\n[t] my  kernel: {my_us:8.1f} us/call", flush=True)
    print(f"[t] cuDNN w13 : {cd_us:8.1f} us/call", flush=True)
    print(f"[t] speedup   : {cd_us / my_us:.2f}x", flush=True)
    print(f"[t] bytes~={bytes_rw:.2f}MB  my_eff_GBps={bytes_rw/(my_us/1e6)/1e3:.1f}", flush=True)


if __name__ == "__main__":
    main()
