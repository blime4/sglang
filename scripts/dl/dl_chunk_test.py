#!/usr/bin/env python
"""Test SGLANG_DL_GDN_DLIN_EXTEND: dl_chunk (fast) vs triton chunk (slow) for GDN prefill.

Measures prefill time (generate(max_new_tokens=1) ≈ prefill) + correctness (greedy output).
Run twice: SGLANG_DL_GDN_DLIN_EXTEND=0 (triton, default) and =1 (dl_chunk).
Compare output_ids (must match for correctness) + prefill time (dl_chunk should be faster).
"""
import os, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
EXTEND = os.environ.get("SGLANG_DL_GDN_DLIN_EXTEND", "0")
PROMPT = os.environ.get("DL_CHUNK_PROMPT",
    "Artificial intelligence is transforming how we work, live, and interact with technology. "
    "From self-driving cars to medical diagnosis, AI systems are becoming increasingly capable "
    "of performing tasks that once required human intelligence. The field encompasses machine "
    "learning, natural language processing, computer vision, and robotics. What are the main"
)

for k, v in {"SGLANG_DL_GDN_DLIN": "1", "SGLANG_DL_MOE_FUSED": "1", "SGLANG_DL_MOE_FUSED_MAX_M": "16",
             "SGLANG_DL_FP8_Q2": "1", "DLEOL_CACHE_SIZE": "1024", "DLEOL_FLA_ENABLE_PINGPONG": "1",
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
    print(f"EXTEND={EXTEND} (0=triton chunk, 1=dl_chunk)", flush=True)
    engine = sgl.Engine(**common)

    for _ in range(3):
        engine.generate(PROMPT, sampling_params={"max_new_tokens": 4, "temperature": 0})

    best_prefill = 999.0
    for _ in range(5):
        t0 = time.perf_counter()
        engine.generate(PROMPT, sampling_params={"max_new_tokens": 1, "temperature": 0})
        best_prefill = min(best_prefill, time.perf_counter() - t0)

    out32 = engine.generate(PROMPT, sampling_params={"max_new_tokens": 32, "temperature": 0})
    text = out32.get("text", "") if isinstance(out32, dict) else getattr(out32, "text", "")
    ids = out32.get("output_ids") if isinstance(out32, dict) else getattr(out32, "output_ids", None)
    engine.shutdown()
    print(f"DL_CHUNK_RESULT extend={EXTEND} prefill_ms={best_prefill*1000:.1f} "
          f"ids={ids} text={text[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
