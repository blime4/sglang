#!/usr/bin/env python3
"""Exp A (cold-prefill JIT artifact) + Exp C (long single-stream decode).

Goal: decompose the SC1 "cold prefill 20.7s" anomaly.
 - Exp A: time prefill of shape#1 (cold, pays dlcc JIT) vs shape#2 (same token-count,
   DIFFERENT content => no RadixAttention/APC cache hit, but kernel already compiled).
   (cold - cold2) ~= one-time dlcc JIT. Also time a true cache-hit (warm) call.
 - Exp C: short prompt + 512-token single-stream decode, best-of-3 tok/s.

One engine per process:  python diag_exp_ac.py --engine sglang|vllm
Cards/TP via env:  CUDA_VISIBLE_DEVICES=24,25,26,27 TP_SIZE=4
"""
import os, sys, time, argparse

# Match the showcase's kernel-path env defaults exactly.
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")

MODEL = os.environ.get("MODEL_PATH",
    "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8")
TP = int(os.environ.get("TP_SIZE", "4"))
MEM = float(os.environ.get("MEM_FRAC", "0.55"))

# Two DISTINCT ~2K-token prefixes (different content => no cache hit; same length => same shape).
PARA_A = ("In distributed inference the scheduler interleaves prefill and decode across "
          "tensor-parallel workers. RadixAttention stores the KV cache in a token-level radix "
          "tree so shared prefixes are computed once and reused across many requests. ")
PARA_B = ("Quantization to eight-bit floating point reduces memory bandwidth pressure on the "
          "mixture-of-experts matmul kernels. The gating network selects two experts per token "
          "and the dispatcher gathers their weights before the fused grouped GEMM executes. ")
NPARA = 30
PREFIX_A = PARA_A * NPARA
PREFIX_B = PARA_B * NPARA
# pad B to identical char length as A so token counts match closely
if len(PREFIX_B) < len(PREFIX_A):
    PREFIX_B += "z" * (len(PREFIX_A) - len(PREFIX_B))


def build(engine):
    if engine == "sglang":
        import sglang as sgl
        e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16",
                       context_length=4096, mem_fraction_static=MEM, max_running_requests=4,
                       disable_cuda_graph=False, cuda_graph_max_bs_decode=4,
                       attention_backend="fa3", page_size=16, disable_custom_all_reduce=True,
                       trust_remote_code=True, chunked_prefill_size=512)
        def gen(prompt, max_new):
            return e.generate(prompt, {"max_new_tokens": max_new, "temperature": 0.0})
        return e, gen, e.shutdown
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16", max_model_len=4096,
                  gpu_memory_utilization=MEM, trust_remote_code=True, max_num_seqs=4,
                  enforce_eager=False, enable_prefix_caching=True,
                  compilation_config={"mode": "none",
                                      "cudagraph_capture_sizes": [1, 2, 4, 528],
                                      "max_cudagraph_capture_size": 528})
        def gen(prompt, max_new):
            sp = SamplingParams(temperature=0.0, max_tokens=max_new)
            return llm.generate([prompt], sp)[0].outputs[0].text
        return llm, gen, lambda: None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()
    print(f"[expAC] engine={args.engine} model={MODEL} tp={TP} mem={MEM} "
          f"cards={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)

    obj, gen, shutdown = build(args.engine)
    print(f"[expAC] {args.engine} engine up.", flush=True)

    if not args.skip_a:
        print("\n[expAC] === Exp A: cold-prefill JIT isolation ===", flush=True)
        t0 = time.perf_counter(); gen(PREFIX_A + "\nSummarize in one sentence.", 8)
        cold = time.perf_counter() - t0
        print(f"[expAC] cold#1 (shape A, JIT paid)        = {cold*1000:8.0f} ms", flush=True)
        t0 = time.perf_counter(); gen(PREFIX_B + "\nSummarize in one sentence.", 8)
        cold2 = time.perf_counter() - t0
        print(f"[expAC] cold#2 (shape B, same len/no hit) = {cold2*1000:8.0f} ms", flush=True)
        t0 = time.perf_counter(); gen(PREFIX_A + "\nSummarize in one sentence.", 8)
        warm = time.perf_counter() - t0
        print(f"[expAC] warm   (shape A again, cache hit) = {warm*1000:8.0f} ms", flush=True)
        jit = cold - cold2
        print(f"[expAC] >>> JIT one-time cost (cold#1-cold#2) ~= {jit*1000:.0f} ms "
              f"({jit/cold*100:.0f}% of cold#1)", flush=True)
        print(f"[expAC] >>> steady prefill+8decode (cold#2) = {cold2*1000:.0f} ms", flush=True)

    if not args.skip_c:
        print("\n[expAC] === Exp C: long single-stream decode (512 tok) ===", flush=True)
        short = "Write a detailed step-by-step plan for deploying a model serving system."
        best = 0.0
        for r in range(3):
            t0 = time.perf_counter(); out = gen(short, 512); dt = time.perf_counter() - t0
            tps = 512.0 / dt
            best = max(best, tps)
            print(f"[expAC] rep{r}: {dt:.2f}s -> {tps:.1f} tok/s", flush=True)
        print(f"[expAC] >>> best 512-tok decode = {best:.1f} tok/s", flush=True)

    shutdown()
    print(f"\n[expAC] DONE engine={args.engine}", flush=True)


if __name__ == "__main__":
    main()
