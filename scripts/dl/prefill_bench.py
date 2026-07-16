#!/usr/bin/env python3
"""OPT-1: prefill 分块绕 dleol。对比 chunked_prefill_size=2048(bf16-bmm 慢) vs 16(fused 快)。

256-tok 长 prompt：max_new_tokens=1 的耗时 ≈ prefill + 1 decode（decode ~恒定，差值≈prefill）。
Env: CHUNK(chunked_prefill_size), FUSED_MAX_M, MODEL, TP_SIZE
"""
import os, sys, time
import sglang
from sglang.srt.server_args import ServerArgs

MODEL = os.environ.get("MODEL", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP_SIZE", "2"))
CHUNK = int(os.environ.get("CHUNK", "16"))
FMAXM = os.environ.get("FUSED_MAX_M", "16")

# ~256-token prompt (repeat). 一个明确 >100 token 的 prompt → baseline(2048) M=256 走 bf16-bmm。
LONG = ("The architecture of modern transformer-based language models relies on self-attention, "
        "feed-forward networks, residual connections, and layer normalization. " * 18)[:1200]

SHORT = "The capital of France is"

def make_engine():
    return sglang.Engine(server_args=ServerArgs(
        model_path=MODEL, dtype="bfloat16", tp_size=TP, attention_backend="fa3",
        page_size=16, mem_fraction_static=0.60, disable_cuda_graph=True, context_length=4096,
        chunked_prefill_size=CHUNK))

def main():
    print(f"CHUNK={CHUNK} FUSED_MAX_M={FMAXM} | LONG prompt ~{len(LONG.split())} words", flush=True)
    e = make_engine()
    # warmup (short, JIT)
    e.generate(SHORT, sampling_params={"max_new_tokens": 4, "temperature": 0})
    e.generate(SHORT, sampling_params={"max_new_tokens": 4, "temperature": 0})

    # SHORT prompt prefill proxy
    t = time.time(); r = e.generate(SHORT, sampling_params={"max_new_tokens": 1, "temperature": 0}); dt = time.time()-t
    print(f"[SHORT] prefill+1tok = {dt*1000:.0f}ms | text: {r['text'][:60]!r}", flush=True)

    # LONG prompt prefill proxy (run 2x for stability)
    for i in range(2):
        t = time.time(); r = e.generate(LONG, sampling_params={"max_new_tokens": 1, "temperature": 0}); dt = time.time()-t
        print(f"[LONG run{i}] prefill+1tok = {dt*1000:.0f}ms | text: {r['text'][:60]!r}", flush=True)

    # LONG prompt + 64 decode (e2e feel)
    t = time.time(); r = e.generate(LONG, sampling_params={"max_new_tokens": 64, "temperature": 0}); dt = time.time()-t
    print(f"[LONG+64dec] total = {dt:.2f}s | text: {r['text'][:80]!r}", flush=True)
    e.shutdown()
    print("[done]", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
