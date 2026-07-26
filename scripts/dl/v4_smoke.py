#!/usr/bin/env python3
# DL: DeepSeek-V4-Flash smoke launch — load + 1 generate, to find the first
# breakage on DLIN (the V4 dsv4 subsystem: compressor/C4-indexer/MTP/sparse-prefill
# + FP8 GEMM + MLA). TP8 (149GB model needs 8x32GB).
#
# Run: source sdk-dlop-07-13-20-30/env.sh
#      CUDA_VISIBLE_DEVICES=16,17,18,19,20,21,22,23 .venv/bin/python scripts/dl/v4_smoke.py
import os, time, sys

for _k, _v in {
    "SGLANG_DL_FP8_Q2": "1", "SGLANG_DL_MOE_FUSED": "1",
    "SGLANG_DL_MOE_FUSED_MAX_M": "2048", "SGLANG_DL_GDN_DLIN": "1",
    "DLEOL_CACHE_SIZE": "1024", "DLEOL_FLA_ENABLE_PINGPONG": "1",
    "DLEOL_FLA_UNROLL_COUNT": "8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(_k, _v)

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))


if __name__ == "__main__":
    import sglang as sgl
    print(f"[v4-smoke] model={MODEL} tp={TP} @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    try:
        engine = sgl.Engine(
            model_path=MODEL, tp_size=TP, dtype="bfloat16",
            trust_remote_code=True, mem_fraction_static=0.90,
            disable_custom_all_reduce=True,  # NCCL (DLIN HC_CUK Error=28 blocker)
            disable_cuda_graph=False,  # DL: try CG (was eager to avoid hang; testing now)
        )
    except Exception as e:
        print(f"[v4-smoke] ENGINE LOAD FAILED ({type(e).__name__}) after "
              f"{time.perf_counter()-t0:.0f}s: {str(e)[:500]}", flush=True)
        import traceback; traceback.print_exc(); sys.exit(1)
    print(f"[v4-smoke] engine up in {time.perf_counter()-t0:.0f}s", flush=True)
    try:
        out = engine.generate("The capital of France is", {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True})
        txt = out["text"] if isinstance(out, dict) else str(out)
        mi = out.get("meta_info", {}) if isinstance(out, dict) else {}
        print(f"[v4-smoke] GENERATED: {txt!r}", flush=True)
        print(f"[v4-smoke] completion_tokens={mi.get('completion_tokens')} "
              f"output_ids={str(mi.get('output_ids',''))[:100]}", flush=True)
        print("[v4-smoke] E2E OK", flush=True)
    except Exception as e:
        print(f"[v4-smoke] GENERATE FAILED ({type(e).__name__}): {str(e)[:500]}", flush=True)
        import traceback; traceback.print_exc(); sys.exit(2)
    engine.shutdown()
