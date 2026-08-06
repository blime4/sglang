#!/usr/bin/env python3
"""DL: V4-Flash in-situ decode profiler — deferred-sync per-component timing to
settle the MoE 17ms-vs-87ms question and locate the real TPOT bottleneck.

Launches V4-Flash TP8 with the blog's serving env + SGLANG_DL_DECODE_PROFILE=1,
warms up, then generates while dl_moe_profile prints running averages every
SGLANG_DL_DECODE_FLUSH calls (default 50):

    [DL_DECODE_PROF] moe_w13=  210.0us/call (n=2150)  moe_w2= 150.0us/call (n=2150)

43 layers x 2 GEMMs/layer = 86 MoE calls/step. If avg*86 ~= 17ms, the blog's
87ms was a sync artifact and the real bottleneck is elsewhere. If ~87ms, the
microbench missed something and the cuDNN op IS the target.

REQUIRES 8 free GPUs. Sync runtime repo first:
  cp python/sglang/srt/layers/quantization/{fp8.py,dl_moe_profile.py} <runtime>/python/sglang/srt/layers/quantization/

Run:
  source .../sdk-0401/env.sh
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8 python scripts/dl/v4_decode_profile.py
"""
import os, time

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))

# Blog V4-Flash serving env (Part 1) + profiler enable.
for _k, _v in {
    "DLI_V2": "ON", "TORCHDYNAMO_DISABLE": "1", "DLEOL_CACHE_SIZE": "1024",
    "SGLANG_DL_MOE_FUSED": "1", "SGLANG_DL_MOE_FUSED_MAX_M": "2048",
    "SGLANG_DL_FP8_Q2": "1", "SGLANG_DL_GDN_DLIN": "1",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "0", "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "0",
    "SGLANG_OPT_USE_TOPK_V2": "0", "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0", "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
    "SGLANG_DL_DECODE_PROFILE": "1",      # enable deferred-sync profiler
    "SGLANG_DL_DECODE_FLUSH": "50",       # print every 50 calls
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(_k, _v)

# Eager by default for clean per-call event attribution (matches blog Part 5).
# Set PROFILE_USE_CG=1 to measure the CG (6.44 tok/s) config instead.
USE_CG = os.environ.get("PROFILE_USE_CG", "0") == "1"

# The deferred-sync profiler calls torch.cuda.synchronize(), which is illegal
# under CUDA-graph capture (invalidates the graph). Only enable it for eager.
if USE_CG:
    os.environ["SGLANG_DL_DECODE_PROFILE"] = "0"

if __name__ == "__main__":
    import sglang as sgl
    print(f"[prof] model={MODEL} tp={TP} CG={USE_CG} @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    engine = sgl.Engine(
        model_path=MODEL, tp_size=TP, dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("PROFILE_MEM_FRAC", "0.90")),
        disable_custom_all_reduce=(os.environ.get("PROFILE_CUSTOM_AR", "0") != "1"), disable_cuda_graph=not USE_CG,
        # CG tuning (match blog: decode-only capture, no prefill CG which is
        # memory-heavy and crashes on 32GB KS38). Capture only M=1 by default.
        disable_prefill_cuda_graph=True,
        cuda_graph_max_bs_decode=int(os.environ.get("PROFILE_CG_MAX_BS", "1")),
        # V4 FP4 weight loading (~4min) + JIT init ≈ 310s exceeds the default 300s
        # watchdog → worker SIGKILL'd (exit -9) mid-init. Give it headroom.
        watchdog_timeout=float(os.environ.get("PROFILE_WATCHDOG", "600")),
        **({"page_size": int(os.environ["PROFILE_PAGE_SIZE"])}
           if os.environ.get("PROFILE_PAGE_SIZE") else {}),
        # context_length: shrinks the indexer page_table capacity (max_c4_seq_len).
        # If the DL op grids on max_c4_seq_len, a small ctx drops idx_logits.
        **({"context_length": int(os.environ["PROFILE_CONTEXT_LEN"])}
           if os.environ.get("PROFILE_CONTEXT_LEN") else {}),
        # KV cache dtype (env-gated). FP4 KV halves KV memory → frees GPU for spec.
        **({"kv_cache_dtype": os.environ["KV_CACHE_DTYPE"]}
           if os.environ.get("KV_CACHE_DTYPE") else {}),
        # Speculative decoding (env-gated). EAGLE uses V4's built-in mtp.0 layer
        # as the draft head (no separate draft model). Enable: SPEC_ALGO=EAGLE.
        **({"speculative_algorithm": os.environ["SPEC_ALGO"],
            "speculative_num_steps": int(os.environ.get("SPEC_NUM_STEPS", "2")),
            "speculative_eagle_topk": int(os.environ.get("SPEC_TOPK", "1")),
            "speculative_num_draft_tokens": int(os.environ.get("SPEC_NUM_DRAFT", "4")),
           } if os.environ.get("SPEC_ALGO") else {}),
    )
    print(f"[prof] engine up in {time.perf_counter()-t0:.0f}s", flush=True)
    # warmup (fills DLEOL JIT cache)
    _w = engine.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0})
    _wt = _w["text"] if isinstance(_w, dict) else str(_w)
    _wi = (_w.get("output_ids") if isinstance(_w, dict) else None)
    print(f"[prof] WARMUP text={_wt!r} ids={_wi}", flush=True)

    # DL: concurrent-requests path — tests whether the model forward amortizes
    # at M=2 (per-request tok/s stays ~M=1 → amortized) or scales (drops). This
    # diagnoses if EAGLE's verify-M=2 gap is model-scaling (fundamental) or
    # framework overhead (fixable → path to 20).
    _conc = int(os.environ.get("PROFILE_CONCURRENT", "0"))
    N = int(os.environ.get("PROFILE_NEW_TOKENS", "256"))
    if _conc > 1:
        prompts = [f"Write a long essay about topic {i}:" for i in range(_conc)]
        t1 = time.perf_counter()
        outs = engine.generate(prompts, {"max_new_tokens": N, "temperature": 0, "ignore_eos": True})
        dt = time.perf_counter() - t1
        total = N * _conc
        print(f"[prof] CONCURRENT M={_conc}: {total} tokens in {dt:.2f}s -> "
              f"{total/dt:.2f} tok/s aggregate, {_conc*N/dt:.2f} per-req-equiv, "
              f"per-req-tok/s={N/dt:.2f} (TPOT {dt/N*1000:.1f}ms)", flush=True)
        engine.shutdown()
        print("[prof] DONE", flush=True)
        import sys; sys.exit(0)

    # timed decode — many steps so deferred-sync flushes multiple times
    t1 = time.perf_counter()
    N = int(os.environ.get("PROFILE_NEW_TOKENS", "256"))
    out = engine.generate("Write a long essay about the history of computing:",
                          {"max_new_tokens": N, "temperature": 0, "ignore_eos": True})
    _txt = out["text"] if isinstance(out, dict) else str(out)
    print(f"[prof] DECODE OUTPUT (first 120): {_txt[:120]!r}", flush=True)
    dt = time.perf_counter() - t1
    print(f"[prof] decoded {N} tokens in {dt:.2f}s -> {N/dt:.2f} tok/s (TPOT {dt/N*1000:.1f}ms)", flush=True)
    engine.shutdown()
    print("[prof] DONE", flush=True)
