#!/usr/bin/env python3
"""Phase-1 diagnostic: what does the TARGET (no MTP) produce greedily?

Resolves the contradiction between:
  - commit a827e8c319: "The Trumps are..." phrase loop (fixed by rep_penalty=1.2)
  - MTP debug notes:     "\\n (token 198) for ALL prompts" (severe)

Different degeneracies = different root causes = different fixes.
Prints token ids so we can see EXACTLY what the target emits greedily vs
greedy+rep_penalty, on the actual MTP test prompt.

Env: PROMPT, REP (default 1.2), TP_SIZE, GPU.
"""
import os
import sys
import time

import sglang

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "Explain how neural networks learn from data.")
REP = float(os.environ.get("REP", "1.2"))
TP = int(os.environ.get("TP_SIZE", "2"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "48"))


def tok_report(r, label):
    text = r["text"]
    ids = r["meta_info"].get("output_ids") or r["meta_info"].get("completion_token_ids") or []
    n_newline = sum(1 for t in ids if t == 198)
    print(f"\n=== {label} ===")
    print(f"text ({len(ids)} toks): {text[:300]!r}")
    print(f"token ids (first 48): {ids[:48]}")
    print(f"#token-198(newline)={n_newline}/{len(ids)}  ratio={n_newline/max(len(ids),1):.2f}")
    # detect phrase-loop: count repeats of first 4-token ngram
    if len(ids) >= 8:
        ngram = tuple(ids[:4])
        reps = sum(1 for i in range(0, len(ids)-3) if tuple(ids[i:i+4]) == ngram)
        print(f"first-4gram repeat count: {reps}")


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
    )
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    print(f"[engine] loaded in {time.time()-t0:.1f}s", flush=True)

    # warmup
    e.generate(PROMPT, sampling_params={"max_new_tokens": 8, "temperature": 0})

    # 1) plain greedy (temp=0, no penalty)
    r1 = e.generate(PROMPT, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
    tok_report(r1, f"PLAIN GREEDY (temp=0, no penalty)")

    # 2) greedy + repetition_penalty
    r2 = e.generate(PROMPT, sampling_params={
        "max_new_tokens": MAX_NEW, "temperature": 0, "repetition_penalty": REP})
    tok_report(r2, f"GREEDY + repetition_penalty={REP}")

    # 3) greedy + frequency_penalty (alternative penalty)
    r3 = e.generate(PROMPT, sampling_params={
        "max_new_tokens": MAX_NEW, "temperature": 0, "frequency_penalty": 0.5})
    tok_report(r3, "GREEDY + frequency_penalty=0.5")

    e.shutdown()
    print("\n[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
