#!/usr/bin/env python3
"""NGRAM speculative-decoding test for Qwen3.5-35B-A3B-FP8 on DLIN.

Goal: measure whether NGRAM spec-decode lifts decode throughput toward 2x vLLM
(25 tok/s). Reports correctness, effective tok/s, and (if exposed) accept rate.

Compare baseline (non-spec) decode 18.32 tok/s from e2e_correctness_speed.py.

Env: PROMPT, MAX_NEW_TOKENS, WARMUP_NEW_TOKENS, TP_SIZE, USE_CUDA_GRAPH,
     NUM_DRAFT (default 4), MIN_BFS (default 1), MAX_BFS (default 1)
"""
import os
import sys
import time
import traceback

import sglang

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "96"))
WARMUP_NEW = int(os.environ.get("WARMUP_NEW_TOKENS", "32"))
TP = int(os.environ.get("TP_SIZE", "2"))
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "4"))
MIN_BFS = int(os.environ.get("MIN_BFS", "1"))
MAX_BFS = int(os.environ.get("MAX_BFS", "1"))


def log(m):
    print(m, flush=True)


def main():
    log(f"[cfg] ngram num_draft={NUM_DRAFT} bfs={MIN_BFS}-{MAX_BFS} "
        f"tp={TP} cg={'on' if os.environ.get('USE_CUDA_GRAPH','0')=='1' else 'off'} "
        f"prompt={PROMPT!r}")

    from sglang.srt.server_args import ServerArgs
    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.80,
        disable_cuda_graph=os.environ.get("USE_CUDA_GRAPH", "0") != "1",
        cuda_graph_max_bs_decode=int(os.environ.get("CG_MAX_BS", "8")),
        context_length=4096,
        speculative_algorithm="NGRAM",
        speculative_num_draft_tokens=NUM_DRAFT,
        speculative_ngram_min_bfs_breadth=MIN_BFS,
        speculative_ngram_max_bfs_breadth=MAX_BFS,
        mamba_track_interval=max(64, NUM_DRAFT),
    )
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    log(f"[engine] loaded in {time.time()-t0:.1f}s")

    # Warmup (triton prefill JIT + spec verify JIT happens here).
    tw = time.time()
    r = e.generate(PROMPT, sampling_params={"max_new_tokens": WARMUP_NEW, "temperature": 0})
    log(f"[warmup] {WARMUP_NEW} tok in {time.time()-tw:.2f}s: {r['text'][:50]!r}")

    # 2 timed runs — effective tok/s (wall-clock / completion_tokens).
    for i in range(2):
        t1 = time.time()
        r = e.generate(PROMPT, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
        dt = time.time() - t1
        n = r["meta_info"]["completion_tokens"]
        meta = r["meta_info"]
        # accept-rate fields vary by version; print any spec-related keys.
        spec_keys = {k: v for k, v in meta.items() if "spec" in k.lower() or "accept" in k.lower() or "draft" in k.lower()}
        log(f"[bench{i}] {n} tok in {dt:.2f}s -> {n/dt:.2f} tok/s effective | spec_meta={spec_keys}")
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
