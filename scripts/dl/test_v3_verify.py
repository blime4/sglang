#!/usr/bin/env python3
"""Verify SGLANG_DL_MOE_V3=1 prefill MoE: (1) diverse correctness, (2) decode
M=1 throughput (must NOT regress — decode uses the old non-v3 path)."""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_GDN_DLIN_EXTEND", "1")
os.environ.setdefault("SGLANG_DL_MOE_V3", "1")

PROMPTS = [
    ("fact", "The capital of France is", 8),
    ("math", "1+1=", 4),
    ("math2", "What is 15 times 4? Answer with just the number.", 8),
    ("english", "Hello, how are you? Tell me about yourself in one sentence.", 30),
    ("chinese", "用中文介绍一下你自己，一句话。", 30),
    ("code", "def fibonacci(n):\n    ", 40),
    ("long", "Write a short paragraph about why the ocean is blue.", 60),
    ("reasoning", "If I have 3 apples and eat 1, then buy 2 more, how many? Think step by step.", 60),
]


def main():
    import statistics, sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    TP = int(os.environ.get("TP_SIZE", "4"))
    print(f"[verify] V3={os.environ.get('SGLANG_DL_MOE_V3')} TP={TP}", flush=True)
    e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True,
        chunked_prefill_size=int(os.environ.get("CHUNK_SIZE", "512")))
    # (1) correctness
    print("\n=== CORRECTNESS ===", flush=True)
    for tag, p, mx in PROMPTS:
        out = e.generate(p, {"max_new_tokens": mx, "temperature": 0})
        txt = out["text"] if isinstance(out, dict) else str(out)
        print(f"[{tag:9}] {txt[:90]!r}", flush=True)
    # (2) decode throughput (SC9-style): short prompt, long generation
    print("\n=== DECODE (M=1 path, must not regress) ===", flush=True)
    sp = {"max_new_tokens": 128, "temperature": 0}
    e.generate("Count from 1 to 5.", sp)  # warmup
    rates = []
    for _ in range(3):
        t0 = time.perf_counter()
        e.generate("Tell me a short story about a robot learning to paint.", sp)
        dt = time.perf_counter() - t0
        rates.append(128 / dt)
    print(f"[decode] 128-token gen: {[f'{r:.1f}' for r in rates]} tok/s | median={statistics.median(rates):.1f}", flush=True)
    # (3) prefill re-confirm
    import numpy as np
    print("\n=== PREFILL (v3 path) ===", flush=True)
    spp = {"max_new_tokens": 1, "temperature": 0}
    e.generate(input_ids=list(range(100, 30000))[:2048], sampling_params=spp)  # warmup
    ts = []
    for s in [1, 2, 3]:
        ids = np.random.default_rng(s).integers(100, 30000, size=2048).tolist()
        t0 = time.perf_counter(); e.generate(input_ids=ids, sampling_params=spp)
        ts.append(time.perf_counter() - t0)
    print(f"[prefill] 2K reps={[round(t*1000) for t in ts]}ms median={statistics.median(ts)*1000:.0f}ms ({2048/statistics.median(ts):.0f} tok/s)", flush=True)
    e.shutdown(); print("[verify] DONE", flush=True)


if __name__ == "__main__":
    main()
