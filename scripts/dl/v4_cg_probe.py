#!/usr/bin/env python3
# DL: DeepSeek-V4-Flash CUDA-graph feasibility probe on DLIN (TP8, 32GB cards).
# Tries CG capture at bs=1 with a low mem_fraction to leave room for the graph
# workspace. Reports OOM / capture error / success so we know if CG is viable.
import os, time, sys
for _k, _v in {"SGLANG_DL_FP8_Q2":"1","SGLANG_DL_MOE_FUSED":"1","SGLANG_DL_MOE_FUSED_MAX_M":"2048",
               "SGLANG_DL_GDN_DLIN":"1","DLEOL_CACHE_SIZE":"1024","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True",
               "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}.items():
    os.environ.setdefault(_k, _v)
MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))
MEM = float(os.environ.get("MEM_FRACTION_STATIC", "0.75"))
if __name__ == "__main__":
    import sglang as sgl
    print(f"[v4-cg] model={MODEL} tp={TP} mem={MEM} CG=on bs_decode=1 @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    try:
        engine = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16",
            trust_remote_code=True, mem_fraction_static=MEM,
            disable_custom_all_reduce=True, disable_cuda_graph=False,
            cuda_graph_max_bs_decode=1, cuda_graph_bs_decode=[1])
    except Exception as e:
        print(f"[v4-cg] ENGINE_INIT_FAIL ({type(e).__name__}) after {time.perf_counter()-t0:.0f}s: {str(e)[:400]}", flush=True)
        import traceback; traceback.print_exc(); sys.exit(2)
    print(f"[v4-cg] engine up (CG captured?) in {time.perf_counter()-t0:.0f}s", flush=True)
    try:
        out = engine.generate("The capital of France is", {"max_new_tokens": 32, "temperature": 0, "ignore_eos": True})
        print(f"[v4-cg] GENERATED: {(out['text'] if isinstance(out,dict) else str(out))[:60]!r}", flush=True)
        print("[v4-cg] CG_OK", flush=True)
    except Exception as e:
        print(f"[v4-cg] GENERATE_FAIL ({type(e).__name__}): {str(e)[:400]}", flush=True)
        import traceback; traceback.print_exc(); sys.exit(3)
    engine.shutdown()
