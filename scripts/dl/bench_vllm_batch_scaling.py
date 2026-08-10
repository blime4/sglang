#!/usr/bin/env python3
# vLLM batch-scaling mirror of bench_batch_scaling.py (sglang), for first-principles
# overhead-vs-compute decomposition. Eager mode (enforce_eager=True) for a clean
# per-sequence compute-slope comparison against sglang's eager 1.57 ms/seq.
import os
import statistics
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16,64").split(",")]
CG = os.environ.get("CUDA_GRAPH", "0") == "1"  # default eager (clean slope compare)
TP = int(os.environ.get("TP", "1"))  # tensor-parallel size (35B FP8 needs TP>=2)
RUNS = int(os.environ.get("RUNS", "5"))
WARMUP = int(os.environ.get("WARMUP_TOKENS", "32"))


def main():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=not CG,
        tensor_parallel_size=TP,
    )
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0, max_tokens=NEW)
    sp_w = SamplingParams(temperature=0, max_tokens=WARMUP)
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] vLLM step_time(batch) (NEW={NEW}, RUNS={RUNS}, min=best)")
    print(f"{'bs':>5} {'tok/s_min':>10} {'step_min':>9} {'step_med':>9} {'step_max':>9}")
    for bs in BATCHES:
        prompts = [prompt] * bs
        for _ in range(2):
            llm.generate(prompts, sp_w)  # warmup
        steps = []
        for _ in range(RUNS):
            t0 = time.time()
            llm.generate(prompts, sp)
            steps.append((time.time() - t0) / NEW * 1000.0)
        smin, smed, smax = min(steps), statistics.median(steps), max(steps)
        tps_min = bs * 1000.0 / smin
        print(f"{bs:>5} {tps_min:>10.1f} {smin:>9.2f} {smed:>9.2f} {smax:>9.2f}")


if __name__ == "__main__":
    main()
