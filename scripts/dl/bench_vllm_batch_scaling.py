#!/usr/bin/env python3
# vLLM batch-scaling mirror of bench_batch_scaling.py (sglang), for first-principles
# overhead-vs-compute decomposition. Eager mode (enforce_eager=True) for a clean
# per-sequence compute-slope comparison against sglang's eager 1.57 ms/seq.
import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16,64").split(",")]
CG = os.environ.get("CUDA_GRAPH", "0") == "1"  # default eager (clean slope compare)


def main():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=not CG,
    )
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0, max_tokens=NEW)
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] vLLM step_time(batch) (NEW={NEW})")
    print(f"{'bs':>5} {'tok/s':>9} {'step_ms':>9}")
    for bs in BATCHES:
        prompts = [prompt] * bs
        llm.generate(prompts, sp)  # warmup
        t0 = time.time()
        llm.generate(prompts, sp)
        dt = time.time() - t0
        step_ms = dt / NEW * 1000.0
        tps = bs * NEW / dt
        print(f"{bs:>5} {tps:>9.1f} {step_ms:>9.2f}")


if __name__ == "__main__":
    main()
