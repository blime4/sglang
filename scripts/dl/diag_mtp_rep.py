#!/usr/bin/env python3
"""Phase-3 diagnostic: does repetition_penalty raise MTP greedy accept?

Phase-1 proved the TARGET's greedy degeneracy is a phrase-repetition loop,
fully fixed by repetition_penalty=1.2 (coherent output). The MTP draft was
trained on the target's coherent (sampled) outputs, so it predicts coherent
content. Hypothesis: in greedy MTP the target degenerates (repeats) ->
mismatch with draft -> 10.6% accept. With rep_penalty the target produces
coherent content matching the draft's training distribution -> accept rises.

Runs ONE MTP engine, two configs, prints accept + text for each.

Env: PROMPT, REP (default 1.2), TP_SIZE, NUM_STEPS, NUM_DRAFT, MAX_NEW_TOKENS
"""
import os
import sys
import time

import sglang

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "Explain how neural networks learn from data.")
REP = float(os.environ.get("REP", "1.2"))
TP = int(os.environ.get("TP_SIZE", "2"))
TOPK = int(os.environ.get("TOPK", "1"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "4"))
# For topk>1 the tree has topk*num_steps+1 nodes; topk=1 chain has num_steps+1.
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", str(TOPK * NUM_STEPS + 1 if TOPK > 1 else NUM_STEPS + 1)))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))


def run(e, label, sp):
    t = time.time()
    r = e.generate(PROMPT, sampling_params=sp)
    dt = time.time() - t
    meta = r["meta_info"]
    n = meta["completion_tokens"]
    spec = {k: v for k, v in meta.items() if "spec" in k.lower() or "accept" in k.lower()}
    print(f"\n=== {label} ===")
    print(f"{n} tok in {dt:.2f}s -> {n/dt:.2f} tok/s effective")
    print(f"spec_meta={spec}")
    print(f"text: {r['text'][:400]!r}")
    return spec


def main():
    from sglang.srt.server_args import ServerArgs
    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.60,
        disable_cuda_graph=True,
        context_length=4096,
        speculative_algorithm="FROZEN_KV_MTP",
        speculative_eagle_topk=TOPK,
        speculative_num_steps=NUM_STEPS,
        speculative_num_draft_tokens=NUM_DRAFT,
        speculative_draft_attention_backend=("triton" if TOPK > 1 else None),
    )
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    print(f"[engine] loaded in {time.time()-t0:.1f}s | steps={NUM_STEPS} draft={NUM_DRAFT}", flush=True)

    # warmup
    e.generate(PROMPT, sampling_params={"max_new_tokens": 16, "temperature": 0})
    e.generate(PROMPT, sampling_params={"max_new_tokens": 16, "temperature": 0,
                                         "repetition_penalty": REP})

    # 1) baseline greedy
    s1 = run(e, f"BASELINE greedy (temp=0, no penalty)",
             {"max_new_tokens": MAX_NEW, "temperature": 0})
    # 2) greedy + rep_penalty
    s2 = run(e, f"GREEDY + repetition_penalty={REP}",
             {"max_new_tokens": MAX_NEW, "temperature": 0, "repetition_penalty": REP})
    # 3) greedy + rep_penalty (repeat for stability)
    s3 = run(e, f"GREEDY + repetition_penalty={REP} (run2)",
             {"max_new_tokens": MAX_NEW, "temperature": 0, "repetition_penalty": REP})
    # 4) sampling temp=0.6 — is the bug greedy-specific or universal?
    s4 = run(e, "SAMPLING temperature=0.6",
             {"max_new_tokens": MAX_NEW, "temperature": 0.6})

    print("\n=== SUMMARY ===")
    print(f"baseline greedy:           {s1}")
    print(f"greedy + rep_penalty={REP}: {s2}")
    print(f"greedy + rep_penalty={REP}: {s3}")
    print(f"sampling temp=0.6:         {s4}")

    e.shutdown()
    print("[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
