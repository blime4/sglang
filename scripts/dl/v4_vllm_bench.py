#!/usr/bin/env python3
"""DL: benchmark vLLM V4-Flash on DLIN KS38 (cards 8-15) to find the sglang gap.
If vLLM gets ~20 tok/s, the gap is unported DL plugin optimizations."""
import os, time

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    print(f"[vllm-bench] model={MODEL} tp={TP} @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
        trust_remote_code=True, max_model_len=4096,
        gpu_memory_utilization=float(os.environ.get("VLLM_MEM_FRAC", "0.90")),
        kv_cache_dtype="fp8",
        enforce_eager=(os.environ.get("VLLM_EAGER", "0") == "1"),
    )
    print(f"[vllm-bench] engine up in {time.perf_counter()-t0:.0f}s", flush=True)
    # warmup
    llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    # timed decode
    N = int(os.environ.get("BENCH_NEW_TOKENS", "256"))
    t1 = time.perf_counter()
    out = llm.generate(["Write a long essay about the history of computing:"],
                       SamplingParams(max_tokens=N, temperature=0, ignore_eos=True))
    dt = time.perf_counter() - t1
    txt = out[0].outputs[0].text if out else ""
    print(f"[vllm-bench] WARMUP+DECODE first 80: {txt[:80]!r}", flush=True)
    print(f"[vllm-bench] decoded {N} tokens in {dt:.2f}s -> {N/dt:.2f} tok/s (TPOT {dt/N*1000:.1f}ms)", flush=True)
    print("[vllm-bench] DONE", flush=True)
