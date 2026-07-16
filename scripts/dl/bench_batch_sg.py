#!/usr/bin/env python3
"""MEAS-1 sglang: 离线批量聚合吞吐 vs batch size（短/长 prompt）。

一次加载，循环 B=1/8/16（+32 若显存允许）。generate([p]*B) 一次批 B 请求。
报：聚合 tok/s（=sum(out)/dt，serving 吞吐代理）、per-req tok/s、总时。
Env: MODEL, TP_SIZE, OUT, CHUNK
"""
import os, sys, time
import sglang
from sglang.srt.server_args import ServerArgs

MODEL = os.environ.get("MODEL", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP_SIZE", "2"))
OUT = int(os.environ.get("OUT", "64"))
CG_ON = os.environ.get("CG", "0") == "1"  # CG=1 enables cuda graph (default eager)
CHUNK = int(os.environ.get("CHUNK", "16"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,8,16").split(",")]

SHORT = "The capital of France is"
LONG = ("The architecture of modern transformer-based language models relies on self-attention, "
        "feed-forward networks, residual connections, and layer normalization. " * 18)[:1200]

def main():
    e = sglang.Engine(server_args=ServerArgs(
        model_path=MODEL, dtype="bfloat16", tp_size=TP, attention_backend="fa3",
        page_size=16, mem_fraction_static=0.60, disable_cuda_graph=not CG_ON, context_length=4096,
        chunked_prefill_size=CHUNK))
    # warmup
    e.generate([SHORT]*2, sampling_params={"max_new_tokens": 16, "temperature": 0})
    e.generate([SHORT]*2, sampling_params={"max_new_tokens": 16, "temperature": 0})

    print(f"\n{'B':>4} {'prompt':>6} {'time_s':>8} {'agg_tok/s':>10} {'per_req_tok/s':>14}", flush=True)
    for B in BATCHES:
        for name, p in [("short", SHORT), ("long", LONG)]:
            prompts = [p] * B
            try:
                t = time.time()
                r = e.generate(prompts, sampling_params={"max_new_tokens": OUT, "temperature": 0, "ignore_eos": True})
                dt = time.time() - t
                toks = sum(ri["meta_info"]["completion_tokens"] for ri in r)
                print(f"{B:>4} {name:>6} {dt:>8.2f} {toks/dt:>10.1f} {toks/B/dt:>14.2f}", flush=True)
            except Exception as ex:
                print(f"{B:>4} {name:>6} ERROR {str(ex)[:80]}", flush=True)
    e.shutdown()
    print("[done]", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
