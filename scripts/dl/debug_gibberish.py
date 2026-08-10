#!/usr/bin/env python3
"""Debug: does Qwen3.5-35B produce correct output on sglang/DLIN?
Decode the actual output token ids for a few prompts."""
import os, sglang
from transformers import AutoTokenizer

def main():
    e = sglang.Engine(
        model_path=os.environ.get("MODEL_PATH", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"),
        dtype="bfloat16", tp_size=int(os.environ.get("TP", "2")),
        attention_backend="fa3", page_size=16, mem_fraction_static=0.80,
        disable_cuda_graph=True, context_length=4096)
    tok = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])

    prompts = [
        "The capital of France is",
        "Hello, how are you? I am",
        "1 + 1 =",
    ]
    for p in prompts:
        r = e.generate(p, sampling_params={"max_new_tokens": 8, "temperature": 0.0})
        ids = r["output_ids"]
        decoded = [tok.decode([i]) for i in ids]
        print(f"\nPROMPT: {p!r}")
        print(f"  output_ids : {ids}")
        print(f"  decoded    : {decoded}")
        print(f"  text       : {r['text']!r}")
    e.shutdown()

if __name__ == "__main__":
    main()
