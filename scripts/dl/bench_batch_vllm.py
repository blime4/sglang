#!/usr/bin/env python3
"""MEAS-1 vLLM: 离线批量聚合吞吐 vs batch size（短/长 prompt）。对照 sglang。

默认 graph on（vLLM 在 DLIN graph 有效）= vLLM 真实 serving 性能。
Env: MODEL, TP, OUT, EAGER(1=enforce_eager,0=graph on), BATCHES
"""
import os, time
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP", "2"))
OUT = int(os.environ.get("OUT", "64"))
EAGER = os.environ.get("EAGER", "0") == "1"
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,8,16").split(",")]

SHORT = "The capital of France is"
LONG = ("The architecture of modern transformer-based language models relies on self-attention, "
        "feed-forward networks, residual connections, and layer normalization. " * 18)[:1200]

def main():
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.60, enforce_eager=EAGER, tensor_parallel_size=TP)
    llm.generate([SHORT]*2, SamplingParams(temperature=0, max_tokens=16))  # warmup
    llm.generate([SHORT]*2, SamplingParams(temperature=0, max_tokens=16))

    print(f"\n{'B':>4} {'prompt':>6} {'time_s':>8} {'agg_tok/s':>10} {'per_req_tok/s':>14}", flush=True)
    for B in BATCHES:
        for name, p in [("short", SHORT), ("long", LONG)]:
            try:
                t = time.time()
                outs = llm.generate([p]*B, SamplingParams(temperature=0, max_tokens=OUT, ignore_eos=True))
                dt = time.time() - t
                toks = sum(len(o.outputs[0].token_ids) for o in outs)
                print(f"{B:>4} {name:>6} {dt:>8.2f} {toks/dt:>10.1f} {toks/B/dt:>14.2f}", flush=True)
            except Exception as ex:
                print(f"{B:>4} {name:>6} ERROR {str(ex)[:80]}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
