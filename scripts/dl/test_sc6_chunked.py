#!/usr/bin/env python3
"""
SC6 根因验证：chunked_prefill_size 2048 vs 512

验证 compare 配置 chunked_prefill_size=512（导致 929-token prompt 被切成 512+417 双 chunk eager）
是否比 serve 默认 chunked_prefill_size=2048（32GB GPU）单 chunk 更慢。
"""
import argparse
import os
import sys
import time
from typing import List, Dict

# HF offline mode (same as compare)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
TP = 4
MEM_FRAC = 0.60  # Same as qwen35-35b preset


def build_unique_prompt(seed: int) -> str:
    """Build a unique ~1K-token prompt to avoid RadixAttention cache hits."""
    base = "The following is a detailed technical report about AI system optimization."
    # Generate distinct content based on seed
    content = f"\n\n[Section {seed}]\n" + "\n".join(
        f"Detail {i}: This is unique content for seed {seed} item {i}. "
        f"Technical parameter alpha_{seed}_{i} = {seed * i % 1000}. "
        f"Configuration beta_{seed}_{i} = {(seed * i * 7) % 1000}."
        for i in range(100)
    )
    return base + content


def run_sc6_raw_prefill(engine, generate_func, prompts: List[str], engine_name: str, chunked_size: int) -> Dict:
    """Run SC6 raw prefill benchmark."""
    print(f"\n=== SC6 Raw Prefill (chunked_prefill_size={chunked_size}) ===", flush=True)
    print(f"Engine: {engine_name}", flush=True)

    # Get token count per prompt
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        tok_per = len(_tok.encode(prompts[0]))
    except Exception as e:
        print(f"Tokenizer load failed: {e}", flush=True)
        tok_per = int(len(prompts[0].split()) * 1.4)

    total_prefill = 8 * tok_per
    print(f"Tokens per prompt: ~{tok_per}, Total prefill per pass: {total_prefill}", flush=True)

    def one_pass(offset):
        t0 = time.perf_counter()
        for k in range(8):
            generate_func(prompts[offset * 8 + k], max_new=4, ignore_eos=True)
        return time.perf_counter() - t0

    # Warmup (JIT)
    print("Warmup pass...", flush=True)
    one_pass(0)

    # Measured passes
    print("Running measured passes...", flush=True)
    times = [one_pass(1), one_pass(2)]
    best = min(times)

    prefill_tps = total_prefill / best
    print(f"SC6 {engine_name} (chunked={chunked_size}): best={best*1000:.0f}ms  "
          f"raw_prefill={prefill_tps:.0f} tok/s  "
          f"reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)

    return {
        "engine": engine_name,
        "chunked_prefill_size": chunked_size,
        "tok_per_prompt": tok_per,
        "best_ms": best * 1000,
        "prefill_tps": prefill_tps,
        "times_ms": [t * 1000 for t in times],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked-prefill-size", type=int, default=2048,
                        help="chunked_prefill_size to test (default: 2048)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs to use (CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"Using GPUs: {args.gpus}", flush=True)

    # Build unique prompts (24 distinct, 3 passes of 8 each)
    pool = [build_unique_prompt(100 + p * 8 + k) for p in range(3) for k in range(8)]
    print(f"Built {len(pool)} unique prompts", flush=True)

    # Import sglang after GPU setting
    import sglang as sgl

    chunked_size = args.chunked_prefill_size
    print(f"\nLaunching sglang engine with chunked_prefill_size={chunked_size}...", flush=True)

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
        chunked_prefill_size=chunked_size,
    )

    def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
        sp = {"max_new_tokens": max_new, "temperature": temperature, "ignore_eos": ignore_eos}
        if n > 1:
            sp["n"] = n
        return engine.generate(prompt, sp)

    # Run SC6
    result = run_sc6_raw_prefill(engine, generate, pool, "sglang", chunked_size)

    print("\n=== RESULT ===", flush=True)
    print(f"chunked_prefill_size={chunked_size}", flush=True)
    print(f"raw_prefill_tps={result['prefill_tps']:.0f} tok/s", flush=True)
    print(f"best_time={result['best_ms']:.0f} ms", flush=True)

    engine.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
