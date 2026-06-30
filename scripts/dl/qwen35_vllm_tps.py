#!/usr/bin/env python3
# Time Qwen3.5-35B-A3B-FP8 decode tok/s on vLLM/DLIN (dl19) — the gap baseline.
import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP", "2"))
N = int(os.environ.get("NTOKENS", "16"))


def main():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        tensor_parallel_size=TP,
    )
    p = ["The capital of France is"]
    llm.generate(p, SamplingParams(temperature=0, max_tokens=4))  # warmup (JIT)
    t0 = time.time()
    llm.generate(p, SamplingParams(temperature=0, max_tokens=N))  # timed decode
    dt = time.time() - t0
    print(f"VLLM_TPS: {N / dt:.2f} tok/s ({N} tokens in {dt:.2f}s)")


if __name__ == "__main__":
    main()
