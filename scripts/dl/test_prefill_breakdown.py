#!/usr/bin/env python3
"""Prefill breakdown: time 2048-token prefill (cache-miss, 3 distinct contents)
under {normal, SGLANG_DL_SKIP_MOE=1, SGLANG_DL_SKIP_ATTN=1}. Run once per mode
(env read in scheduler subprocess). Reveals whether MoE, attention, or the
residual (GDN/Mamba + norm) dominates the ~41s prefill.
"""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
MODE = os.environ.get("BREAKDOWN_MODE", "normal")


def main():
    import numpy as np
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    TP = int(os.environ.get("TP_SIZE", "4"))
    print(f"[breakdown] MODE={MODE} SKIP_MOE={os.environ.get('SGLANG_DL_SKIP_MOE','0')} "
          f"SKIP_ATTN={os.environ.get('SGLANG_DL_SKIP_ATTN','0')}", flush=True)
    e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
    sp = {"max_new_tokens": 1, "temperature": 0}
    times = []
    for seed in [1, 2, 3]:  # distinct content -> all cache misses
        ids = np.random.default_rng(seed).integers(100, 30000, size=2048).tolist()
        t0 = time.perf_counter(); e.generate(input_ids=ids, sampling_params=sp)
        times.append(time.perf_counter() - t0)
        print(f"[breakdown] {MODE} 2048 seed{seed} = {times[-1]*1000:.0f} ms", flush=True)
    import statistics
    print(f"[breakdown] {MODE} MEDIAN = {statistics.median(times)*1000:.0f} ms "
          f"({2048/statistics.median(times):.0f} tok/s)", flush=True)
    e.shutdown(); print("[breakdown] DONE", flush=True)


if __name__ == "__main__":
    main()
