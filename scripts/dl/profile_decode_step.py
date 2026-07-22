"""Profile a single decode step of Qwen3.5-35B on DLIN.

Measures the REAL breakdown: # of kernel launches, CPU vs CUDA time,
top kernels by time. Uses DLPTI_AUTO_LOAD for DLIN-native profiling
(torch.profiler alone crashes on DLIN).

Must have `if __name__ == '__main__':` guard — sglang uses multiprocessing
spawn for TP workers, which re-imports the main module.
"""
import os, sys, time, torch
from pathlib import Path

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
TP = int(os.environ.get("TP", "4"))
MEM_FRAC = os.environ.get("MEM_FRAC", "0.82")
CTX = os.environ.get("CONTEXT_LEN", "4096")
MRR = os.environ.get("MAX_RUNNING_REQUESTS", "16")
N_PROFILE = int(os.environ.get("N_PROFILE", "3"))

os.environ.setdefault("SGLANG_DL_MOE_DLBLAS", "1")
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "32")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_MULTI_STEP", "1")
os.environ.setdefault("DLEOL_CACHE_SIZE", "1024")


def main():
    import sglang as sgl

    engine = sgl.Engine(
        model_path=MODEL,
        tp_size=TP,
        dtype="bfloat16",
        context_length=int(CTX),
        mem_fraction_static=float(MEM_FRAC),
        max_running_requests=int(MRR),
        disable_cuda_graph=True,
        attention_backend="fa3",
        disable_custom_all_reduce=True,
        trust_remote_code=True,
    )
    print(f"[profile] engine up.", flush=True)

    warm_prompt = "Explain the theory of relativity in detail."
    for _ in range(2):
        engine.generate(warm_prompt, {"max_new_tokens": 16, "temperature": 0.0})
    print(f"[profile] warmup done.", flush=True)

    prof_prompt = "Write a long essay about artificial intelligence."

    import torch.profiler as prof
    with prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as p:
        r = engine.generate(prof_prompt, {"max_new_tokens": N_PROFILE, "temperature": 0.0})

    print(f"[profile] generated {N_PROFILE} tokens. Analyzing...", flush=True)

    trace_path = "/tmp/sglang_decode_trace.json"
    p.export_chrome_trace(trace_path)
    print(f"[profile] trace → {trace_path}", flush=True)

    print("\n" + "=" * 70)
    print("TOP 25 CUDA KERNELS BY TOTAL TIME (self_time)")
    print("=" * 70)
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

    print("\n" + "=" * 70)
    print("TOP 25 OPS BY CPU TIME (includes dispatch)")
    print("=" * 70)
    print(p.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    ka = p.key_averages()
    total_cuda = sum(e.self_cuda_time_total for e in ka)
    total_cpu = sum(e.self_cpu_time_total for e in ka)
    total_count = sum(e.count for e in ka)
    n_unique = len(ka)

    print("\n" + "=" * 70)
    print(f"OVER {N_PROFILE} DECODE STEP(S):")
    print(f"  total unique ops:    {n_unique}")
    print(f"  total op instances:  {total_count}")
    print(f"  total CUDA time:     {total_cuda/1000:.1f} ms")
    print(f"  total CPU time:      {total_cpu/1000:.1f} ms")
    print(f"  CUDA time/step:      {total_cuda/N_PROFILE/1000:.1f} ms")
    print(f"  CPU time/step:       {total_cpu/N_PROFILE/1000:.1f} ms")
    print(f"  ops/step:            {total_count/N_PROFILE:.0f}")
    print("=" * 70)

    engine.shutdown()


if __name__ == "__main__":
    main()
