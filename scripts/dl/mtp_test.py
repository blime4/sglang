#!/usr/bin/env python3
"""MTP (Frozen-KV MTP) speculative-decoding test for Qwen3.5-35B-A3B-FP8 on DLIN.

The checkpoint ships a native 1-layer MTP head (`mtp.*` weights). sglang exposes
it as speculative_algorithm="FROZEN_KV_MTP" (draft reads target KV, no draft KV).
Qwen3.5 frozen-KV hooks were ported from gemma4 (build_frozen_kv_mtp_context +
bind_frozen_kv_context in qwen3_5_mtp.py; save_kv_cache guard in qwen3_5.py).

Compare vs NGRAM (scripts/dl/ngram_test.py): MTP adds a 1-layer draft forward per
step but uses a *learned* draft (higher accept on diverse text where n-gram fails).

Env: PROMPT, MAX_NEW_TOKENS, TP_SIZE, NUM_STEPS (draft depth), NUM_DRAFT (total draft)
"""
import os
import sys
import time
import traceback

import sglang

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "Explain how neural networks learn from data.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TP = int(os.environ.get("TP_SIZE", "4"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "4"))
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", str(NUM_STEPS + 1)))
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.60"))


def log(m):
    print(m, flush=True)


def main():
    log(f"[cfg] FROZEN_KV_MTP num_steps={NUM_STEPS} num_draft={NUM_DRAFT} eagle_topk=1 "
        f"tp={TP} cg=off(draft) prompt={PROMPT!r}")

    from sglang.srt.server_args import ServerArgs
    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=MEM_FRAC,
        disable_cuda_graph=True,  # draft CG off on DLIN initially (per MTP report)
        context_length=4096,
        disable_custom_all_reduce=True,  # DL: TP>1 on DLIN needs NCCL (HC_CUK Error=28)
        speculative_algorithm="FROZEN_KV_MTP",
        speculative_eagle_topk=1,
        speculative_num_steps=NUM_STEPS,
        speculative_num_draft_tokens=NUM_DRAFT,
    )
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    log(f"[engine] loaded in {time.time()-t0:.1f}s")

    # warmup (draft JIT + verify JIT)
    tw = time.time()
    r = e.generate(PROMPT, sampling_params={"max_new_tokens": 16, "temperature": 0})
    log(f"[warmup] 16 tok in {time.time()-tw:.2f}s: {r['text'][:50]!r}")

    for i in range(2):
        t1 = time.time()
        r = e.generate(PROMPT, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
        dt = time.time() - t1
        n = r["meta_info"]["completion_tokens"]
        meta = r["meta_info"]
        spec = {k: v for k, v in meta.items() if "spec" in k.lower() or "accept" in k.lower()}
        log(f"[bench{i}] {n} tok in {dt:.2f}s -> {n/dt:.2f} tok/s effective | spec_meta={spec}")
    log(f"[OUT-START]{r['text']}[OUT-END]")

    e.shutdown()
    log("[done]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
