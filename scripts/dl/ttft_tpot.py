#!/usr/bin/env python3
"""Measure TTFT (time to first token) and TPOT (time per output token) for
sglang on DLIN via the streaming API. Single-stream, in-process (OFFLINE).

Env: PROMPT, MAX_NEW_TOKENS, TP_SIZE, USE_CUDA_GRAPH, CG_MAX_BS,
     SGLANG_DL_MOE_FUSED*, and (NGRAM) USE_NGRAM/NGRAM_NUM_DRAFT/...
"""
import os
import time

import sglang

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TP = int(os.environ.get("TP_SIZE", "2"))
USE_NGRAM = os.environ.get("USE_NGRAM", "0") == "1"
ND = int(os.environ.get("NGRAM_NUM_DRAFT", "8"))


def log(m):
    print(m, flush=True)


def main():
    from sglang.srt.server_args import ServerArgs
    sa = ServerArgs(
        model_path=MODEL, dtype="bfloat16", tp_size=TP, attention_backend="fa3",
        page_size=16, mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.60")),
        disable_cuda_graph=os.environ.get("USE_CUDA_GRAPH", "0") != "1",
        cuda_graph_max_bs_decode=int(os.environ.get("CG_MAX_BS", "2")),
        context_length=4096,
        # DL: custom all-reduce (cross_device_reduce_1stage<bfloat16,2>) can't
        # JIT on DLIN (HC_CUK Error=28) under TP>1 + CG → use NCCL. See docs §7.x.
        disable_custom_all_reduce=os.environ.get("DL_DISABLE_CUSTOM_AR", "1") == "1",
        # DL: torch.compile the model before CG capture (vLLM uses inductor-driven
        # CG; tests whether compile makes invoke_fused_moe_opt's use_moe_cu kernel
        # capturable — see docs §7.31). Opt-in via DL_TORCH_COMPILE=1.
        enable_torch_compile=os.environ.get("DL_TORCH_COMPILE", "0") == "1",
        # DL: tc_piecewise decode backend (MoE as split-op → eager, may avoid
        # use_moe_cu CG crash). Opt-in via DL_CG_BACKEND_DECODE=tc_piecewise.
        cuda_graph_backend_decode=os.environ.get("DL_CG_BACKEND_DECODE") or None,
    )
    if USE_NGRAM:
        sa.speculative_algorithm = "NGRAM"
        sa.speculative_num_draft_tokens = ND
        sa.speculative_ngram_min_bfs_breadth = 1
        sa.speculative_ngram_max_bfs_breadth = 1
        sa.mamba_track_interval = max(64, ND)
    e = sglang.Engine(server_args=sa)
    # warmup (JIT)
    e.generate(PROMPT, sampling_params={"max_new_tokens": 8, "temperature": 0})

    # Streaming measurement: time first token (TTFT) and each subsequent (ITL/TPOT).
    sp = {"max_new_tokens": MAX_NEW, "temperature": 0}
    t0 = time.time()
    stream = e.generate(PROMPT, sampling_params=sp, stream=True)
    ttft = None
    itls = []
    n_tok = 0
    for chunk in stream:
        text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        if text:
            now = time.time()
            if ttft is None:
                ttft = now - t0  # first token
            else:
                itls.append(now - last)
            last = now
            n_tok += len(text.split()) or 1  # rough; tokens arrive in chunks
    total = time.time() - t0
    # completion_tokens is the authoritative count; fall back to counting chunks
    tag = f"NGRAM(num_draft={ND})" if USE_NGRAM else "plain(fused)"
    log(f"[{tag}] TTFT={ttft*1000:.1f} ms | total={total:.2f}s")
    if itls:
        import statistics
        log(f"  inter-token latency (TPOT proxy): median={statistics.median(itls)*1000:.1f} ms "
            f"mean={statistics.mean(itls)*1000:.1f} ms over {len(itls)} chunks")
    log(f"  wall-clock: {MAX_NEW/total:.2f} tok/s (requested {MAX_NEW} tok)")
    e.shutdown()


if __name__ == "__main__":
    main()
