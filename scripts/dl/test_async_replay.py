#!/usr/bin/env python3
"""
async-replay 量化：decode gap 改善测试

测试 SGLANG_DL_ASYNC_REPLAY=1 对纯 decode 性能的影响。
使用短 prompt + 较长 decode 来测量 decode-bound 场景。
"""
import argparse
import os
import sys
import time
from typing import Dict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
TP = 4
MEM_FRAC = 0.60


def run_decode_benchmark(engine, generate_func, prompt: str, max_new: int, num_iters: int = 5) -> Dict:
    """Run pure decode benchmark (short prompt)."""
    print(f"\n=== Decode Benchmark: prompt+{max_new} tokens, {num_iters} iters ===", flush=True)

    # Warmup
    print("Warmup...", flush=True)
    generate_func(prompt, max_new=max_new, temperature=0.0)

    # Measured runs
    latencies = []
    total_tokens = 0
    for i in range(num_iters):
        t0 = time.perf_counter()
        output = generate_func(prompt, max_new=max_new, temperature=0.0)
        t1 = time.perf_counter()

        # Count output tokens (rough estimate)
        if isinstance(output, dict):
            text = output.get("text", "")
        elif isinstance(output, list) and len(output) > 0:
            text = output[0].get("text", "") if isinstance(output[0], dict) else str(output[0])
        else:
            text = str(output)

        # Rough token count (1.4 chars per token estimate)
        tokens = int(len(text) * 1.4)
        total_tokens += tokens

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        print(f"  iter {i+1}: {latency_ms:.1f}ms, ~{tokens} tokens", flush=True)

    # Compute stats
    avg_latency = sum(latencies) / len(latencies)
    best_latency = min(latencies)
    total_time = sum(lat / 1000 for lat in latencies)

    avg_tps = total_tokens / total_time
    best_tps = total_tokens / (best_latency / 1000)

    print(f"\nResults (avg over {num_iters}):", flush=True)
    print(f"  avg TPOT: {avg_latency:.1f}ms", flush=True)
    print(f"  best TPOT: {best_latency:.1f}ms", flush=True)
    print(f"  avg tok/s: {avg_tps:.1f}", flush=True)
    print(f"  best tok/s: {best_tps:.1f}", flush=True)

    return {
        "avg_tpot_ms": avg_latency,
        "best_tpot_ms": best_latency,
        "avg_tps": avg_tps,
        "best_tps": best_tps,
        "latencies": latencies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-replay", action="store_true", help="Enable async replay")
    parser.add_argument("--time-replay", type=int, default=0, help="Enable timing replay (1=with sync, 2=wall-clock)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs")
    parser.add_argument("--max-new", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--iters", type=int, default=5, help="Number of iterations")
    args = parser.parse_args()

    # Set env vars
    if args.async_replay:
        os.environ["SGLANG_DL_ASYNC_REPLAY"] = "1"
    if args.time_replay > 0:
        os.environ["SGLANG_DL_TIME_REPLAY"] = str(args.time_replay)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    mode = "async-replay" if args.async_replay else "baseline"
    print(f"Mode: {mode}", flush=True)
    print(f"GPUs: {args.gpus}", flush=True)

    # Import sglang after GPU setting
    import sglang as sgl

    print(f"\nLaunching sglang engine...", flush=True)

    engine = sgl.Engine(
        model_path=MODEL,
        tp_size=args.tp,
        dtype="bfloat16",
        context_length=4096,
        mem_fraction_static=MEM_FRAC,
        max_running_requests=4,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4,
        attention_backend="fa3",
        page_size=16,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        chunked_prefill_size=512,
    )

    def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
        sp = {"max_new_tokens": max_new, "temperature": temperature, "ignore_eos": ignore_eos}
        if n > 1:
            sp["n"] = n
        return engine.generate(prompt, sp)

    # Use a short prompt for decode-bound testing
    prompt = "Explain the concept of machine learning in detail:"

    # Run benchmark
    result = run_decode_benchmark(engine, generate, prompt, args.max_new, args.iters)

    print("\n=== FINAL RESULT ===", flush=True)
    print(f"mode={mode}", flush=True)
    print(f"best_tpot_ms={result['best_tpot_ms']:.1f}", flush=True)
    print(f"best_tps={result['best_tps']:.1f}", flush=True)
    print(f"avg_tpot_ms={result['avg_tpot_ms']:.1f}", flush=True)
    print(f"avg_tps={result['avg_tps']:.1f}", flush=True)

    engine.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
