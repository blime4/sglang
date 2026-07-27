#!/usr/bin/env python3
"""Validate: does warming M=chunked_prefill_size(512) kill the prefill JIT?
Uses input_ids for EXACT token counts. chunked_prefill_size=512.
 - warm 512 (the full-chunk M every long prefill hits).
 - time prefill 2048 (=4x512, all full chunks) -> should be STEADY if 512 warmed.
 - time prefill 2000 (=3x512 + 464) -> full chunks warm, but partial M=464 may JIT.
 - warm 464, re-time 2000 -> if fast now, confirms partial-M JIT is the only residual.
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
    def tpref(n, seed):
        t0 = time.perf_counter(); e.generate(input_ids=ids(n, seed), sampling_params=sp)
        return time.perf_counter() - t0
    def twarm(n, seed):
        t0 = time.perf_counter(); e.generate(input_ids=ids(n, seed), sampling_params=sp)
        print(f"[warmup-test] warm M={n} took {time.perf_counter()-t0:.1f}s", flush=True)

    print("[warmup-test] === A) NO warmup: cold prefill 2048 (4x512) ===", flush=True)
    print(f"  cold 2048 = {tpref(2048, 1)*1000:.0f} ms", flush=True)
    print("[warmup-test] === B) warm M=512 (full-chunk), then re-measure ===", flush=True)
    twarm(512, 7)
    print(f"  2048 (4x512, warmed)  = {tpref(2048, 2)*1000:.0f} ms", flush=True)
    print(f"  2000 (3x512 + 464)    = {tpref(2000, 3)*1000:.0f} ms  <- partial M=464", flush=True)
    print("[warmup-test] === C) also warm M=464, re-measure 2000 ===", flush=True)
    twarm(464, 8)
    print(f"  2000 (512+464 warmed) = {tpref(2000, 4)*1000:.0f} ms", flush=True)
    e.shutdown(); print("[warmup-test] DONE", flush=True)


if __name__ == "__main__":
    main()
