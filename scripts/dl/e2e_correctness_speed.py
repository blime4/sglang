#!/usr/bin/env python3
"""E2E correctness + decode-speed test for Qwen3.5-35B-A3B-FP8 on DLIN.

Runs a deterministic prompt, prints the FULL output text, and measures
steady-state decode tok/s. All output is flushed so background polling works.

Env:
  PROMPT, MAX_NEW_TOKENS, WARMUP_NEW_TOKENS, TP_SIZE,
  CUDA_VISIBLE_DEVICES (set externally), SGLANG_DL_MOE_FUSED, SGLANG_DL_MOE_MAX_BF16_M
"""
import os
import sys
import time
import traceback

import sglang


MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
WARMUP_NEW = int(os.environ.get("WARMUP_NEW_TOKENS", "32"))
TP = int(os.environ.get("TP_SIZE", "2"))


def log(msg):
    print(msg, flush=True)


def main():
    log(f"[cfg] fused={os.environ.get('SGLANG_DL_MOE_FUSED','?')} "
        f"max_bf16_m={os.environ.get('SGLANG_DL_MOE_MAX_BF16_M','?')} "
        f"tp={TP} prompt={PROMPT!r}")

    from sglang.srt.server_args import ServerArgs
    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.80,
        disable_cuda_graph=os.environ.get("USE_CUDA_GRAPH", "0") != "1",
        cuda_graph_max_bs_decode=int(os.environ.get("CG_MAX_BS", "1")),
        context_length=4096,
    )
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    log(f"[engine] loaded in {time.time()-t0:.1f}s")

    # Warmup (triton prefill JIT happens here for this unique M).
    log(f"[warmup] {WARMUP_NEW} tok...")
    tw = time.time()
    r = e.generate(PROMPT, sampling_params={"max_new_tokens": WARMUP_NEW, "temperature": 0})
    dw = time.time() - tw
    log(f"[warmup] {WARMUP_NEW} tok in {dw:.2f}s ({WARMUP_NEW/dw:.2f} tok/s incl prefill JIT): {r['text'][:50]!r}")

    # 3 timed decode runs to measure STEADY STATE (JIT amortized).
    sp = {"max_new_tokens": MAX_NEW, "temperature": 0}
    for i in range(3):
        t1 = time.time()
        r = e.generate(PROMPT, sampling_params=sp)
        dt = time.time() - t1
        n_tok = r["meta_info"]["completion_tokens"]
        log(f"[bench{i}] {n_tok} tok in {dt:.2f}s -> {n_tok/dt:.2f} tok/s")
    log(f"[OUT-START]{r['text']}[OUT-END]")

    e.shutdown()
    log("[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
