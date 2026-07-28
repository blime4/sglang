#!/usr/bin/env python3
"""Test NO_BREAK CG with chunked_prefill_size=2048 (1 chunk for 2K prefill).
Hypothesis: 1 cudaGraphLaunch instead of 4 saves ~2700ms of launch overhead.
Previous eager chunk=2048: 5649ms (363 tps). With CG, Python dispatch eliminated."""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_BCG_NO_BREAK", "1")

def main():
    import numpy as np
    import statistics
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    CS = int(os.environ.get("CHUNK_SIZE", "2048"))
    print(f"[cg-chunk] chunked_prefill_size={CS} + NO_BREAK CG", flush=True)
    e = sgl.Engine(model_path=MODEL, tp_size=4, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=CS,
        cuda_graph_backend_prefill="breakable")
    sp = {"max_new_tokens": 1, "temperature": 0}
    # correctness check
    out = e.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0})
    txt = out["text"] if isinstance(out, dict) else str(out)
    print(f"[cg-chunk] correctness: 'capital of France' -> {txt[:60]!r}", flush=True)
    # warmup
    e.generate(input_ids=list(range(100, 30000))[:2048], sampling_params=sp)
    # timing: 3 distinct 2K prefills
    ts = []
    for s in [1, 2, 3]:
        ids = np.random.default_rng(s).integers(100, 30000, size=2048).tolist()
        t0 = time.perf_counter(); e.generate(input_ids=ids, sampling_params=sp)
        ts.append(time.perf_counter() - t0)
    print(f"[cg-chunk] CS={CS} 2K reps={[round(t*1000) for t in ts]}ms "
          f"median={statistics.median(ts)*1000:.0f}ms ({2048/statistics.median(ts):.0f} tok/s)", flush=True)
    e.shutdown(); print("[cg-chunk] DONE", flush=True)

if __name__ == "__main__":
    main()
