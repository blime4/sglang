#!/usr/bin/env python3
# Layer-level decode timing for Qwen3.5-35B linear_attn vs mlp split.
import os, time, triton, triton.language.extra.cuda as _tlc

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


def main():
    e = sglang.Engine(
        model_path=MODEL, dtype="bfloat16", tp_size=2,
        attention_backend="fa3", page_size=16, mem_fraction_static=0.82,
        disable_cuda_graph=True, context_length=4096, max_running_requests=16,
    )
    p = ["The capital of France is"]
    e.generate(p, sampling_params={"max_new_tokens": 16, "temperature": 0})
    # decode 1 token with timing enabled
    os.system("rm -f /tmp/dl_layer_timing.csv")
    t0 = time.time()
    e.generate(p, sampling_params={"max_new_tokens": 1, "temperature": 0})
    print(f"TOTAL decode step: {(time.time() - t0) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
