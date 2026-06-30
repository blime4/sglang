#!/usr/bin/env python3
# First-principles perf decomposition: measure step_time(batch) for sglang decode,
# then fit  step_time = a + b*batch  -> a = fixed per-step overhead (engine
# round-trip + launch), b = per-sequence compute (asymptote -> memory-BW floor).
# torch.profiler crashes on DLIN, so we decompose via batch-scaling instead.
#
# One engine instance tests all batch sizes (amortize model load + JIT).
# Eager mode (disable_cuda_graph) for a clean overhead vs compute split.
#
# NOTE: Engine creation MUST be under `if __name__ == "__main__"` — sglang spawns
# the scheduler subprocess via spawn, which re-imports this module; an unguarded
# top-level Engine() recursively respawns the scheduler and dies at init.
import os
import time

import sglang

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16,64").split(",")]
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.88"))
CG = os.environ.get("CUDA_GRAPH", "0") == "1"
CG_MAX_BS = int(os.environ.get("CG_MAX_BS", "0"))  # cuda_graph_max_bs_decode (0=default)


def main():
    kw = dict(
        model_path=MODEL,
        page_size=16,
        dtype="bfloat16",
        attention_backend=BACKEND,
        disable_cuda_graph=not CG,
        mem_fraction_static=MEM_FRAC,
    )
    if CG and CG_MAX_BS:
        kw["cuda_graph_max_bs_decode"] = CG_MAX_BS
    engine = sglang.Engine(**kw)
    PROMPT = "The capital of France is"

    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] step_time(batch) decomposition (NEW={NEW} tokens/seq)")
    print(f"{'bs':>5} {'tok/s':>9} {'step_ms':>9}")
    for bs in BATCHES:
        prompts = [PROMPT] * bs
        engine.generate(prompts, sampling_params={"max_new_tokens": 8})  # warmup
        t0 = time.time()
        engine.generate(
            prompts, sampling_params={"max_new_tokens": NEW, "temperature": 0}
        )
        dt = time.time() - t0
        step_ms = dt / NEW * 1000.0
        tps = bs * NEW / dt
        print(f"{bs:>5} {tps:>9.1f} {step_ms:>9.2f}")


if __name__ == "__main__":
    main()
