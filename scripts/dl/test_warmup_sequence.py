#!/usr/bin/env python3
"""Decisive: does a warmup SEQUENCE (128,256,512,1024) make a 2048 prefill fast,
and is there an input_ids vs string difference? Reconciles sweep(2.3s) vs
single-shape-warm(41s) discrepancy.
"""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")


def main():
    import numpy as np
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    TP = int(os.environ.get("TP_SIZE", "4"))
    e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
    sp = {"max_new_tokens": 1, "temperature": 0}
    def ids(n, seed): return (np.random.default_rng(seed).integers(100, 30000, size=n)).tolist()
    def t_ids(n, seed):
        t0 = time.perf_counter(); e.generate(input_ids=ids(n, seed), sampling_params=sp); return time.perf_counter()-t0

    print(f"[wseq] cold 2048 input_ids (1st call) = {t_ids(2048,1)*1000:.0f} ms", flush=True)
    print("[wseq] warmup SEQUENCE (untimed): 128,256,512,1024 ...", flush=True)
    for n in [128, 256, 512, 1024]:
        t0=time.perf_counter(); e.generate(input_ids=ids(n,99), sampling_params=sp)
        print(f"   warm {n} = {time.perf_counter()-t0:.1f}s", flush=True)
    b1 = min(t_ids(2048,2), t_ids(2048,3))
    print(f"[wseq] 2048 input_ids AFTER sequence (best-of-2) = {b1*1000:.0f} ms", flush=True)
    e.shutdown(); print("[wseq] DONE", flush=True)


if __name__ == "__main__":
    main()
