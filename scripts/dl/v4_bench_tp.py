#!/usr/bin/env python3
# DL: DeepSeek-V4-Flash decode TPOT benchmark on DLIN (eager, TP8).
# Measures per-token decode time after a warmup, separated from prefill.
import os, time, sys
for _k, _v in {"SGLANG_DL_FP8_Q2":"1","SGLANG_DL_MOE_FUSED":"1","SGLANG_DL_MOE_FUSED_MAX_M":"2048",
               "SGLANG_DL_GDN_DLIN":"1","DLEOL_CACHE_SIZE":"1024","DLEOL_FLA_ENABLE_PINGPONG":"1",
               "DLEOL_FLA_UNROLL_COUNT":"8","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
               "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}.items():
    os.environ.setdefault(_k, _v)
MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))
WARMUP = int(os.environ.get("WARMUP_TOKENS", "32"))
MEASURE = int(os.environ.get("MEASURE_TOKENS", "128"))
USE_CG = os.environ.get("USE_CG", "0") == "1"
CG_BACKEND = os.environ.get("CG_BACKEND", "breakable")  # breakable | full
MEM = float(os.environ.get("MEM_FRACTION_STATIC", "0.90"))
CTX = int(os.environ.get("CONTEXT_LENGTH", "0"))  # 0 = default
if __name__ == "__main__":
    import sglang as sgl
    print(f"[v4-bench] model={MODEL} tp={TP} CG={USE_CG} backend={CG_BACKEND} mem={MEM} ctx={CTX} @ {time.strftime('%H:%M:%S')}", flush=True)
    kw = dict(model_path=MODEL, tp_size=TP, dtype="bfloat16",
        trust_remote_code=True, mem_fraction_static=MEM,
        disable_custom_all_reduce=True, disable_cuda_graph=not USE_CG)
    if USE_CG:
        kw.update(cuda_graph_max_bs_decode=1, cuda_graph_bs_decode=[1],
                  cuda_graph_backend_decode=CG_BACKEND)
    if CTX > 0:
        kw.update(context_length=CTX)
    t0 = time.perf_counter()
    engine = sgl.Engine(**kw)
    print(f"[v4-bench] engine up in {time.perf_counter()-t0:.0f}s", flush=True)
    # warmup (drives any lazy JIT — silu/MoE kernels)
    engine.generate("The capital of France is", {"max_new_tokens": WARMUP, "temperature": 0, "ignore_eos": True})
    # measured decode run
    t0 = time.perf_counter()
    out = engine.generate("The capital of France is", {"max_new_tokens": MEASURE, "temperature": 0, "ignore_eos": True})
    dt = time.perf_counter() - t0
    mi = out.get("meta_info", {}) if isinstance(out, dict) else {}
    n_tok = mi.get("completion_tokens", MEASURE)
    tpot_ms = dt / max(n_tok, 1) * 1000
    _txt = out["text"] if isinstance(out, dict) else str(out)
    print(f"[v4-bench] GENERATED({n_tok}tok): {_txt[:60]!r}", flush=True)
    print(f"[v4-bench] TPOT={tpot_ms:.1f}ms  tok/s={1000/tpot_ms:.2f}  (wall {dt:.1f}s for {n_tok} tok, incl prefill)", flush=True)
    engine.shutdown()
