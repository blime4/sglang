#!/usr/bin/env python3
"""Test SGLANG_DL_GDN_DLIN_EXTEND=1: does routing prefill GDN to the DLIN dl_chunk
kernel (instead of the default triton chunk) fix the ~41s prefill AND stay correct?
"""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_GDN_DLIN_EXTEND", "1")  # THE FLAG UNDER TEST


def main():
    import numpy as np
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    TP = int(os.environ.get("TP_SIZE", "4"))
    print(f"[gdn-ext] GDN_DLIN_EXTEND={os.environ.get('SGLANG_DL_GDN_DLIN_EXTEND')}", flush=True)
    e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
    sp = {"max_new_tokens": 1, "temperature": 0}
    # correctness
    out = e.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0})
    txt = out["text"] if isinstance(out, dict) else str(out)
    print(f"[gdn-ext] CORRECTNESS 'capital of France' -> {txt!r}", flush=True)
    # timing: 3 distinct 2048 prefills
    import statistics
    times = []
    for seed in [1, 2, 3]:
        ids = np.random.default_rng(seed).integers(100, 30000, size=2048).tolist()
        t0 = time.perf_counter(); e.generate(input_ids=ids, sampling_params=sp)
        times.append(time.perf_counter() - t0)
    print(f"[gdn-ext] 2048 prefill times: {[f'{t*1000:.0f}' for t in times]} ms "
          f"| MEDIAN={statistics.median(times)*1000:.0f}ms ({2048/statistics.median(times):.0f} tok/s)", flush=True)
    e.shutdown(); print("[gdn-ext] DONE", flush=True)


if __name__ == "__main__":
    main()
