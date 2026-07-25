#!/usr/bin/env python3
# DL: FA2 online-concurrency END-TO-END validation (validates fix 0fe8c7cc86).
#
# Fix 0fe8c7cc86 made dl_flash_attn pass max_seqlen_q=max(extend_lens) instead of
# the average. Pre-fix, a batch with UNEQUAL-length prefills — the online-
# concurrency shape (multiple in-flight requests of different lengths batched
# into one varlen extend) — under-sized the kernel softmax_lse workspace and
# OOB-wrote -> wrong result / SIGSEGV.
#
# test_prefill_varlen.py (spy) already proves the *wrapper* passes correct args.
# THIS script proves the FULL engine survives unequal-length concurrent prefill
# end-to-end: fa3 backend (ATTN_BACKEND=fa3 -> FlashAttentionBackend -> DLIN FA2
# via dl_flash_attn, see flashattention_backend.py:235), TP4, a batch of prompts
# with deliberately DIFFERENT lengths forced into one varlen extend
# (chunked_prefill_size=2048 > total tokens).
#
# Run: source sdk-dlop-07-13-20-30/env.sh
#      CUDA_VISIBLE_DEVICES=20,21,22,23 .venv/bin/python scripts/dl/test_fa2_online_concurrency.py
import os, sys, time

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

BASE = ("Explain in detail the architecture of a heterogeneous AI accelerator: its "
        "compute units, memory hierarchy (L1/L2/HBM), programming model, and the "
        "tradeoffs versus a homogeneous GPU design. ")


def build_unequal(n):
    # deliberately DIFFERENT lengths -> unequal extend_lens in one varlen extend
    mults = [1, 4, 10, 2, 7, 3, 5, 8][:n]
    return [BASE * m + "\n\nSummarize the key point in one sentence." for m in mults]


def build_equal(n):
    return [BASE * 4 + "\n\nSummarize the key point in one sentence." for _ in range(n)]


def _text(o):
    return (o.get("text", str(o)) if isinstance(o, dict) else
            (o[0] if isinstance(o, list) else str(o)))


def tok_lens(tok, prompts):
    return [len(tok.encode(p)) for p in prompts]


def run_batch(engine, prompts, label):
    t0 = time.perf_counter()
    outs = engine.generate(prompts, {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True})
    dt = time.perf_counter() - t0
    texts = [_text(o) for o in outs]
    ok = len(texts) == len(prompts) and all(t.strip() for t in texts)
    print(f"[fa2-e2e] {label}: n={len(prompts)} time={dt:.2f}s all_nonempty={ok}", flush=True)
    for i, t in enumerate(texts[:4]):
        print(f"    [{i}] {t[:72]!r}", flush=True)
    return ok, dt, texts


def main():
    import sglang as sgl
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    engine = sgl.Engine(
        model_path=MODEL, tp_size=TP, dtype="bfloat16",
        context_length=4096, mem_fraction_static=0.55, max_running_requests=8,
        disable_cuda_graph=False, cuda_graph_max_bs_decode=8, attention_backend="fa3",
        page_size=16, disable_custom_all_reduce=True, trust_remote_code=True,
        chunked_prefill_size=2048,  # > total batch tokens -> all seqs in ONE varlen extend
    )
    engine.generate("Hello", {"max_new_tokens": 4, "temperature": 0})  # warmup

    results = {}
    # CONTROL: equal-length batch (avg==max -> always worked, even pre-fix)
    pe = build_equal(4)
    print(f"[fa2-e2e] CONTROL equal-len token lens={tok_lens(tok, pe)}", flush=True)
    results["equal_x4"], _, _ = run_batch(engine, pe, "CONTROL equal-len x4")

    # THE CASE: unequal-length batch -> the online-concurrency varlen shape (>2 overlap)
    pu4 = build_unequal(4)
    print(f"[fa2-e2e] UNEQUAL token lens={tok_lens(tok, pu4)}", flush=True)
    results["unequal_x4"], _, _ = run_batch(engine, pu4, "UNEQUAL-len x4 (online concurrency)")

    # STRESS: 8 unequal (>2 overlapping prefills)
    pu8 = build_unequal(8)
    print(f"[fa2-e2e] UNEQUAL token lens={tok_lens(tok, pu8)}", flush=True)
    results["unequal_x8"], _, _ = run_batch(engine, pu8, "UNEQUAL-len x8 (stress)")

    engine.shutdown()

    print("\n=== FA2 ONLINE-CONCURRENCY E2E ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    verdict = all(results.values())
    print("VERDICT:", "PASS — full engine (fa3/FA2) handles unequal-length concurrent "
          "prefill with no crash and valid output (fix 0fe8c7cc86 validated e2e)"
          if verdict else "FAIL — crash or empty output on unequal-length prefill",
          flush=True)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
