#!/usr/bin/env python3
"""Prefill-length scaling probe: how does prefill tok/s scale with prompt length?
 - If sglang tok/s is FLAT across lengths (~same as vLLM's lower bound) => per-token
   compute-bound (GDN/Mamba scan or MoE per-token).
 - If sglang tok/s DROPS at large length => large-M path / chunking issue.
Each length uses DISTINCT content (no cache hit). best-of-2. 8-token decode appended.
One engine per process: --engine sglang|vllm
"""
import os, time, argparse
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
MODEL = os.environ.get("MODEL_PATH",
    "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8")
TP = int(os.environ.get("TP_SIZE", "4")); MEM = float(os.environ.get("MEM_FRAC", "0.55"))

# 4 distinct passages so each length has unique content (no RadixAttention/APC hit)
PASSAGES = [
    "Distributed inference batches prefill and decode across tensor-parallel workers. ",
    "Quantization to eight-bit float cuts weight memory for the mixture-of-experts GEMM. ",
    "The radix tree indexes KV cache tokens so shared prefixes compute once and reuse. ",
    "A gated recurrent network scans its state sequentially across every input position. ",
]

def build_prompt(target_tokens, idx):
    per = len(PASSAGES[idx % 4].split())  # ~12 tokens
    reps = max(1, target_tokens // per)
    return (PASSAGES[idx % 4] * reps) + "\nSummarize in one sentence."

def build(engine):
    if engine == "sglang":
        import sglang as sgl
        e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
            mem_fraction_static=MEM, max_running_requests=4, disable_cuda_graph=False,
            cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
            disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
        def gen(p): return e.generate(p, {"max_new_tokens": 8, "temperature": 0.0})
        return e, gen, e.shutdown
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16", max_model_len=4096,
            gpu_memory_utilization=MEM, trust_remote_code=True, max_num_seqs=4, enforce_eager=False,
            enable_prefix_caching=True,
            compilation_config={"mode":"none","cudagraph_capture_sizes":[1,2,4,528],"max_cudagraph_capture_size":528})
        def gen(p): return llm.generate([p], SamplingParams(temperature=0.0, max_tokens=8))[0].outputs[0].text
        return llm, gen, lambda: None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True, choices=["sglang","vllm"])
    args = ap.parse_args()
    print(f"[sweep] engine={args.engine} tp={TP} mem={MEM} cards={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    obj, gen, shutdown = build(args.engine)
    print(f"[sweep] {args.engine} up. prefill-length sweep (distinct content, best-of-2):", flush=True)
    print(f"{'length':>8} {'best_ms':>10} {'tok/s':>8}", flush=True)
    for i, L in enumerate([128, 512, 1024, 2048]):
        p = build_prompt(L, i)
        best = 9e9
        for _ in range(2):
            t0 = time.perf_counter(); gen(p); best = min(best, time.perf_counter() - t0)
        print(f"{L:>8} {best*1000:>10.0f} {L/best:>8.1f}", flush=True)
    shutdown(); print(f"[sweep] DONE {args.engine}", flush=True)

if __name__ == "__main__": main()
