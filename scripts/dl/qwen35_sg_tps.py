#!/usr/bin/env python3
# Time Qwen3.5-35B-A3B-FP8 decode tok/s on sglang/DLIN — gap measurement (sglang side).
import os
import time

import triton
import triton.language.extra.cuda as _tlc

# DL: Hopper PDL extras absent from DLIN Triton (FLA kernels reference them in AST).
if not hasattr(_tlc, "gdc_wait"):
    @triton.jit
    def _w():
        pass
    _tlc.gdc_wait = _w
if not hasattr(_tlc, "gdc_launch_dependents"):
    @triton.jit
    def _d():
        pass
    _tlc.gdc_launch_dependents = _d

import sglang

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP", "2"))
N = int(os.environ.get("NTOKENS", "4"))  # short — sglang decode is very slow


def main():
    e = sglang.Engine(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.82,
        disable_cuda_graph=True,
    )
    p = ["The capital of France is"]
    e.generate(p, sampling_params={"max_new_tokens": 2, "temperature": 0})  # warmup (JIT)
    t0 = time.time()
    e.generate(p, sampling_params={"max_new_tokens": N, "temperature": 0})  # timed decode
    dt = time.time() - t0
    print(f"SG_TPS: {N / dt:.4f} tok/s ({N} tokens in {dt:.1f}s)")


if __name__ == "__main__":
    main()
