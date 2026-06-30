#!/usr/bin/env python3
# vLLM decode tok/s benchmark on DLIN — mirror of bench_decode_tps.py (sglang).
# Same model, same prompt, same greedy sampling. Run with SDK env + venv-vllm-bench.
import os, time
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
RUNS = int(os.environ.get("RUNS", "3"))
CG = os.environ.get("CUDA_GRAPH", "1") == "1"  # vLLM default: cuda graph ON

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
    # warmup
    llm.generate([prompt], sp)
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}]")
    print(f"{'run':>4} {'tokens':>7} {'wall_s':>8} {'tok/s':>8}")
    for r in range(RUNS):
        t0 = time.time()
        out = llm.generate([prompt], sp)
        dt = time.time() - t0
        text = out[0].outputs[0].text
        print(f"{r:>4} {NEW:>7} {dt:>8.3f} {NEW/dt:>8.2f}")
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] prompt: '{prompt}' -> '{text[:50]}'")


if __name__ == "__main__":
    main()
