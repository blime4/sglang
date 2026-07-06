#!/usr/bin/env python3
# Time Qwen3.5-35B-A3B-FP8 decode tok/s on sglang/DLIN — gap measurement (sglang side).
import os
import time

import sglang

MODEL = os.environ.get("MODEL_PATH", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/")
TP = int(os.environ.get("TP_SIZE", os.environ.get("TP", "2")))
N = int(os.environ.get("NTOKENS", "4"))
CG = os.environ.get("USE_CUDA_GRAPH", os.environ.get("CUDA_GRAPH", "0")) == "1"
CG_MAX_BS = int(os.environ.get("CG_MAX_BS", "0"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", N))


def main():
    kw = dict(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=int(os.environ.get("PAGE_SIZE", "16")),
        mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.80")),
        disable_cuda_graph=not CG,
        context_length=int(os.environ.get("CONTEXT_LEN", "4096")),
        max_running_requests=int(os.environ.get("MAX_RUNNING_REQUESTS", "0")) or None,
    )
    if CG and CG_MAX_BS:
        kw["cuda_graph_max_bs_decode"] = CG_MAX_BS
    e = sglang.Engine(**kw)
    e.generate(PROMPT, sampling_params={"max_new_tokens": 2, "temperature": 0})  # warmup
    t0 = time.time()
    r = e.generate(PROMPT, sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0})
    dt = time.time() - t0
    print(f"SG_TPS[cuda_graph={CG}]: {MAX_NEW_TOKENS / dt:.4f} tok/s ({MAX_NEW_TOKENS} tokens in {dt:.1f}s)")
    print(f"OUT: {r['text'][:60]!r}")
    e.shutdown()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
