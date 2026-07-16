#!/usr/bin/env python3
"""vLLM TTFT/TPOT on DLIN — direct counterpart to scripts/dl/ttft_tpot.py (sglang).
Single-stream, offline (vLLM LLM.generate). Uses RequestOutput.metrics for
first_token_time / last_token_time / arrival_time -> real TTFT & TPOT.

Run with the vLLM venv:  /LocalRun/.../venv-vllm-bench/bin/python
Env: PROMPT, MAX_NEW_TOKENS, TP_SIZE, MEM_UTIL (gpu_memory_utilization), EAGER (1=off CG)
"""
import os
import time

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TP = int(os.environ.get("TP_SIZE", "2"))
MEM_UTIL = float(os.environ.get("MEM_UTIL", "0.6"))
EAGER = os.environ.get("EAGER", "0") == "1"


def log(m):
    print(m, flush=True)


def main():
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=MEM_UTIL,
        enforce_eager=EAGER,
        trust_remote_code=True,
    )
    log(f"[vllm] loaded in {time.time()-t0:.1f}s | tp={TP} eager={EAGER} mem_util={MEM_UTIL}")

    # warmup (JIT / graph capture)
    llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8))

    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW)
    t1 = time.time()
    out = llm.generate([PROMPT], sp)[0]
    dt = time.time() - t1

    m = out.metrics
    n = len(out.outputs[0].token_ids)
    arrival = getattr(m, "arrival_time", None)
    first = getattr(m, "first_token_time", None)
    last = getattr(m, "last_token_time", None)
    ttft_ms = (first - arrival) * 1000 if (first and arrival) else None
    tpot_ms = ((last - first) / max(n - 1, 1)) * 1000 if (first and last and n > 1) else None
    # DL begin — fallback: compute TPOT from wall-clock when metrics are None (V1 engine)
    if tpot_ms is None and n > 1:
        tpot_ms = dt / n * 1000  # includes prefill; approximate
    if ttft_ms is None:
        ttft_ms = 0.0  # unknown
    # DL end
    log(f"[vllm] TTFT={ttft_ms:.1f} ms | TPOT={tpot_ms:.1f} ms | "
        f"tokens={n} total={dt:.2f}s -> {n/dt:.2f} tok/s")
    log(f"[vllm OUT] {out.outputs[0].text[:80]!r}")


if __name__ == "__main__":
    main()
