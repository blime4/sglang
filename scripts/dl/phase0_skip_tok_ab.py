#!/usr/bin/env python3
# DL: Phase 0 falsification for the in-process-detokenizer plan.
#
# Question: does removing the DETOKENIZER hop improve SC9 decode THROUGHPUT?
# skip_tokenizer_init repoints the scheduler's output socket straight to the
# TokenizerManager (ipc_channels.py:55-59), bypassing the DetokenizerManager —
# i.e. it removes the detokenizer hop (an UPPER BOUND on what in-process detok
# can gain, since it also skips detok CPU entirely).
#
# Decision (criterion #3): if skip mode improves SC9 tok/s by >=5% -> the detok
# hop is a throughput lever -> proceed to build in-process detokenizer. If <5%
# -> the hop is NOT the lever (the 6.2ms is the scheduler<->tokenizer DISPATCH
# round-trip per the 7/22 profiling blog) -> pivot to inline-scheduler, don't
# build in-process detokenizer.
#
# skip_tokenizer_init needs input_ids (it raises on text prompts), so tokenize
# client-side. Run twice: A normal (text), B skip (input_ids). Compare tok/s.
#
#   CUDA_VISIBLE_DEVICES=20,21,22,23 .venv/bin/python scripts/dl/phase0_skip_tok_ab.py   # A
#   CUDA_VISIBLE_DEVICES=20,21,22,23 DL_SKIP_TOK_INIT=1 .venv/bin/python scripts/dl/phase0_skip_tok_ab.py  # B
import os, time

for _k, _v in {
    "SGLANG_DL_FP8_Q2": "1", "SGLANG_DL_MOE_FUSED": "1",
    "SGLANG_DL_MOE_FUSED_MAX_M": "2048", "SGLANG_DL_GDN_DLIN": "1",
    "DLEOL_CACHE_SIZE": "1024", "DLEOL_FLA_ENABLE_PINGPONG": "1",
    "DLEOL_FLA_UNROLL_COUNT": "8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(_k, _v)

MODEL = os.environ.get(
    "MODEL_PATH",
    "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8",
)
TP = int(os.environ.get("TP_SIZE", "4"))
SKIP = bool(int(os.environ.get("DL_SKIP_TOK_INIT", "0")))
PROMPT = ("Write a detailed technical essay about the future of heterogeneous "
          "AI accelerators and their software stacks.")
MAX_NEW = 128

if __name__ == "__main__":
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok.encode(PROMPT)
    engine = sgl.Engine(
        model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512,
        skip_tokenizer_init=SKIP,
    )
    print(f"[phase0] skip_tokenizer_init={SKIP} (mode {'B: hop removed' if SKIP else 'A: baseline'}) "
          f"prompt_tok={len(ids)} max_new={MAX_NEW}", flush=True)

    def gen(n):
        sp = {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}
        if SKIP:
            return engine.generate(sampling_params=sp, input_ids=ids)
        return engine.generate(PROMPT, sp)

    for _ in range(3):  # warmup the decode CG path
        gen(16)
    best = 9e9
    for _ in range(3):
        t0 = time.perf_counter()
        gen(MAX_NEW)
        best = min(best, time.perf_counter() - t0)
    tps = MAX_NEW / best
    print(f"[phase0] RESULT skip_tokenizer_init={SKIP} best={best*1000:.0f}ms "
          f"tps={tps:.2f} tok/s (TPOT={best/MAX_NEW*1000:.2f}ms)", flush=True)
    engine.shutdown()
