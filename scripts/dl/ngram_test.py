#!/usr/bin/env python
"""NGRAM spec decoding test for Qwen3.5-35B-A3B on DLIN. Zero draft cost."""
import os, time

def main():
    for k, v in {"SGLANG_DL_GDN_DLIN":"1","SGLANG_DL_MOE_FUSED":"1","SGLANG_DL_MOE_FUSED_MAX_M":"16",
                 "SGLANG_DL_FP8_Q2":"1","DLEOL_CACHE_SIZE":"1024","DLEOL_FLA_ENABLE_PINGPONG":"1",
                 "DLEOL_FLA_UNROLL_COUNT":"8","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}.items():
        os.environ.setdefault(k, v)
    import sglang as sgl
    PROMPT = os.environ.get("NGRAM_PROMPT",
        "Explain the concept of machine learning in three sentences. Focus on how models learn from data.")
    engine = sgl.Engine(model_path="/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/", dtype="bfloat16",
        tp_size=4, attention_backend="fa3", page_size=16, context_length=4096,
        mem_fraction_static=0.60, disable_cuda_graph=False, disable_custom_all_reduce=True,
        trust_remote_code=True, speculative_algorithm="NGRAM", speculative_num_draft_tokens=8)
    for _ in range(3):
        engine.generate(PROMPT, sampling_params={"max_new_tokens":4,"temperature":0})
    best=999.0
    for _ in range(3):
        t0=time.perf_counter()
        out=engine.generate(PROMPT, sampling_params={"max_new_tokens":64,"temperature":0})
        best=min(best,time.perf_counter()-t0)
    ids=out.get("output_ids") if isinstance(out,dict) else getattr(out,"output_ids",None)
    engine.shutdown()
    print(f"[ngram] TPOT={best/64*1000:.2f}ms tps={64/best:.1f} ids={ids} "
          f"text={(out.get('text','') if isinstance(out,dict) else getattr(out,'text',''))[:60]!r}", flush=True)

if __name__ == "__main__":
    main()
