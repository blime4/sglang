#!/usr/bin/env python
"""MTP (Frozen-KV MTP) correctness + perf probe for Qwen3.5-35B-A3B-FP8 on DLIN.

The model has a native MTP head (1560 mtp.* weights, mtp_num_hidden_layers=1).
MTP in sglang = FROZEN_KV_MTP (draft layer reads target's KV cache, no own KV).

Usage:
  CUDA_VISIBLE_DEVICES=20,21,22,23 python scripts/dl/mtp_correctness.py plain
  CUDA_VISIBLE_DEVICES=20,21,22,23 python scripts/dl/mtp_correctness.py mtp
"""
import os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
assert MODE in ("plain", "mtp"), f"unknown mode {MODE}"
PROMPT = os.environ.get("MTP_PROMPT",
    "Explain the concept of machine learning in three sentences. "
    "Focus on how models learn from data.")
MAX_NEW = int(os.environ.get("MTP_MAX_NEW", "64"))

for k, v in {"SGLANG_DL_GDN_DLIN": "1", "SGLANG_DL_MOE_FUSED": "1",
             "SGLANG_DL_MOE_FUSED_MAX_M": "16", "SGLANG_DL_FP8_Q2": "1",
             "DLEOL_CACHE_SIZE": "1024", "DLEOL_FLA_ENABLE_PINGPONG": "1",
             "DLEOL_FLA_UNROLL_COUNT": "8"}.items():
    os.environ.setdefault(k, v)


def main():
    import sglang as sgl

    common = dict(
        model_path="/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/",
        dtype="bfloat16", tp_size=4, attention_backend="fa3", page_size=16,
        context_length=4096, mem_fraction_static=0.60,
        disable_cuda_graph=False, disable_custom_all_reduce=True,
        trust_remote_code=True,
    )
    if MODE == "mtp":
        common["speculative_algorithm"] = "FROZEN_KV_MTP"
        common["speculative_num_steps"] = int(os.environ.get("MTP_NUM_STEPS", "3"))
        common["speculative_num_draft_tokens"] = common["speculative_num_steps"] + 1
        common["speculative_eagle_topk"] = 1
        # Limit CG capture sizes to avoid OOM on 32GB DLIN cards (default
        # max_running_requests=48 → 21 capture batches × 3.3GB = OOM).
        common["max_running_requests"] = 4
        common["cuda_graph_max_bs_decode"] = 8

    print(f"[{MODE}] config: {MODE}", flush=True)
    engine = sgl.Engine(**common)
    print(f"[{MODE}] engine built", flush=True)

    for _ in range(3):
        engine.generate(PROMPT, sampling_params={"max_new_tokens": 4, "temperature": 0})

    best = 999.0
    for _ in range(3):
        t0 = time.perf_counter()
        out = engine.generate(PROMPT, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
        dt = time.perf_counter() - t0
        best = min(best, dt)

    text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
    ids = out.get("output_ids") if isinstance(out, dict) else getattr(out, "output_ids", None)
    engine.shutdown()
    tpot = best / MAX_NEW * 1000
    print(f"[{MODE}] TPOT={tpot:.2f}ms tps={MAX_NEW/best:.1f} ids={ids} text={text[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
