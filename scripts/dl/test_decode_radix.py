#!/usr/bin/env python3
"""SC9 decode: radix-cache ON vs OFF. For single-stream pure decode, RadixAttention's
per-step tree mgmt is pure overhead (no reuse). Tests if --disable-radix-cache speeds decode.
NO_RADIX=0 (default) | NO_RADIX=1 . best-of-3, 512 tok, ignore_eos.
"""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_GDN_DLIN_EXTEND", "1")


def main():
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    no_radix = os.environ.get("NO_RADIX", "0") == "1"
    e = sgl.Engine(model_path=MODEL, tp_size=4, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512,
        disable_radix_cache=no_radix)
    sp = {"max_new_tokens": 512, "temperature": 0.0, "ignore_eos": True}
    e.generate("Warmup hello world plan deploy system.", sp)
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        e.generate("Write a detailed step-by-step plan for deploying a model serving system.", sp)
        reps.append(512 / (time.perf_counter() - t0))
    import statistics
    print(f"[decode] NO_RADIX={no_radix} -> 5 reps (tok/s): {[round(r,1) for r in reps]} "
          f"| mean={statistics.mean(reps):.1f} min={min(reps):.1f} max={max(reps):.1f}", flush=True)
    e.shutdown(); print("[decode] DONE", flush=True)


if __name__ == "__main__":
    main()
