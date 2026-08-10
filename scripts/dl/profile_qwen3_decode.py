#!/usr/bin/env python3
# Profile Qwen3-1.7B DECODE forward steps on DLIN (in-process sglang.Engine).
#
# Why in-process: the HTTP `serve` path crashes on DLIN (ModuleNotFoundError:
# flashinfer at launch_server import), so sglang.test.send_one --profile (which
# needs the HTTP server) cannot be used. This script drives the Engine directly
# and arms the scheduler-subprocess profiler via engine.start_profile().
#
# The profiler runs inside the scheduler subprocess (where CUDA kernels execute),
# so this captures real GPU kernel timing -- a main-process torch.profiler
# wrapper would see nothing.
#
# Env (optional): MODEL_PATH, ATTN_BACKEND, NUM_STEPS, OUTPUT_DIR, PROMPT.
# Assumes the clean DLIN runtime env is active (run_sglang.sh sets it).
import os
import time
import glob

import sglang

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")
NUM_STEPS = int(os.environ.get("NUM_STEPS", "15"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/qwen3_dlin_decode_profile")
PROMPT = os.environ.get("PROMPT", "Write a short poem about the ocean.")


def main():
    engine = sglang.Engine(
        model_path=MODEL,
        page_size=16,            # DLIN FA2 verified at page_size=16
        dtype="bfloat16",
        attention_backend=BACKEND,
        disable_cuda_graph=True,  # DLIN default: keep off for the baseline profile
    )

    # Warmup: triggers FA2 JIT compilation of ~42 buckets (slow first time).
    print("[warmup] generating to JIT-compile attention buckets ...", flush=True)
    engine.generate([PROMPT], sampling_params={"max_new_tokens": 8})
    print("[warmup] done\n", flush=True)

    # Arm profiler in by_stage mode: non-blocking, profiles first NUM_STEPS
    # DECODE forward steps, then auto-stops + dumps a chrome trace to OUTPUT_DIR.
    print(f"[profile] arming profiler: num_steps={NUM_STEPS}, by_stage=decode", flush=True)
    engine.start_profile(
        output_dir=OUTPUT_DIR,
        num_steps=NUM_STEPS,
        activities=["CPU", "GPU"],
        profile_by_stage=True,
    )

    # Generate enough decode tokens to cover the profiled window.
    out = engine.generate(
        [PROMPT], sampling_params={"max_new_tokens": NUM_STEPS + 4}
    )
    results = out if isinstance(out, list) else [out]
    text = results[0]["text"] if isinstance(results[0], dict) else str(results[0])
    print(f"\n[OUTPUT ] {PROMPT}{text}\n", flush=True)

    # Give the scheduler a moment to flush the trace, then locate it.
    time.sleep(3)
    try:
        engine.stop_profile()
    except Exception as e:
        print(f"[profile] stop_profile note: {e}", flush=True)

    traces = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "*.trace.json*"), recursive=True))
    print(f"[profile] output_dir: {OUTPUT_DIR}", flush=True)
    print(f"[profile] trace files: {traces}", flush=True)
    if not traces:
        # the profiler may nest under a timestamp subdir
        all_json = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "*.json*"), recursive=True))
        print(f"[profile] all json under output_dir: {all_json}", flush=True)


if __name__ == "__main__":
    main()
