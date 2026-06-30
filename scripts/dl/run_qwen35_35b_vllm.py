#!/usr/bin/env python3
# Bring-up + minimal generate for Qwen3.5-35B-A3B-FP8 on vLLM/DLIN (dl19).
# Comparison baseline for the sglang-vs-vLLM gap. vLLM's DL platform plugin adapts
# FP8 (dlblas) + MoE + linear-attn. TP=2 (35B FP8 ≈ 37GB).
import os

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP", "2"))


def main():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # eager for bring-up (correctness first)
        tensor_parallel_size=TP,
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=16),
    )
    print("\n===== GENERATED =====")
    print(repr(out[0].outputs[0].text))
    print("===== DONE =====")


if __name__ == "__main__":
    main()
