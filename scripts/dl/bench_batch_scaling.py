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
import statistics
import time

# DL: DLIN Triton lacks Hopper PDL extras gdc_wait/gdc_launch_dependents (FLA linear-attn
# kernels reference them in the AST; USE_GDC=False at runtime but AST hash needs them to
# resolve). Empty @triton.jit no-ops satisfy the hash + compile to nothing.
import triton
import triton.language.extra.cuda as _tlc

if not hasattr(_tlc, "gdc_wait"):
    @triton.jit
    def _gdc_wait():
        pass
    _tlc.gdc_wait = _gdc_wait
if not hasattr(_tlc, "gdc_launch_dependents"):
    @triton.jit
    def _gdc_launch_dependents():
        pass
    _tlc.gdc_launch_dependents = _gdc_launch_dependents

import sglang

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16,64").split(",")]
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.88"))
CG = os.environ.get("CUDA_GRAPH", "0") == "1"
CG_MAX_BS = int(os.environ.get("CG_MAX_BS", "0"))  # cuda_graph_max_bs_decode (0=default)
TP = int(os.environ.get("TP", "1"))  # tensor-parallel size (35B FP8 needs TP>=2)
RUNS = int(os.environ.get("RUNS", "5"))  # timed runs per batch (min=least contention)
WARMUP = int(os.environ.get("WARMUP_TOKENS", "32"))


def main():
    kw = dict(
        model_path=MODEL,
        page_size=16,
        dtype="bfloat16",
        attention_backend=BACKEND,
        disable_cuda_graph=not CG,
        mem_fraction_static=MEM_FRAC,
        tp_size=TP,
    )
    if CG and CG_MAX_BS:
        kw["cuda_graph_max_bs_decode"] = CG_MAX_BS
    engine = sglang.Engine(**kw)
    PROMPT = "The capital of France is"

    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] step_time(batch) (NEW={NEW}, RUNS={RUNS}, min=best/least-contention)")
    print(f"{'bs':>5} {'tok/s_min':>10} {'step_min':>9} {'step_med':>9} {'step_max':>9}")
    for bs in BATCHES:
        prompts = [PROMPT] * bs
        for _ in range(2):  # thorough warmup (JIT + cache) — MUST match timed sampling
            engine.generate(
                prompts, sampling_params={"max_new_tokens": WARMUP, "temperature": 0}
            )
        steps = []
        for _ in range(RUNS):
            t0 = time.time()
            engine.generate(
                prompts, sampling_params={"max_new_tokens": NEW, "temperature": 0}
            )
            steps.append((time.time() - t0) / NEW * 1000.0)
        smin, smed, smax = min(steps), statistics.median(steps), max(steps)
        tps_min = bs * 1000.0 / smin
        print(f"{bs:>5} {tps_min:>10.1f} {smin:>9.2f} {smed:>9.2f} {smax:>9.2f}")


if __name__ == "__main__":
    main()
