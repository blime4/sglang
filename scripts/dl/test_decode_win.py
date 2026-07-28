#!/usr/bin/env python3
"""Decode-win sweep: single-stream decode at 256/512/1024 tokens, sglang vs vLLM.
Decode is sglang's structural advantage (no prefill, no GDN-flag needed). Confirm the
win holds and how the margin scales with length (IPC amortization).
One engine per process: --engine sglang|vllm. best-of-3, short prompt.
"""
import os, time, argparse
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    args = ap.parse_args()
    TP = int(os.environ.get("TP_SIZE", "4")); MEM = 0.55
    print(f"[decode] engine={args.engine} tp={TP} cards={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
    short = "Write a detailed step-by-step plan for deploying a distributed model serving system."
    if args.engine == "sglang":
        import sglang as sgl
        e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
            mem_fraction_static=MEM, max_running_requests=4, disable_cuda_graph=False,
            cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
            disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
        def gen(n): return e.generate(short, {"max_new_tokens": n, "temperature": 0.0, "ignore_eos": True})
        shut = e.shutdown
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16", max_model_len=4096,
            gpu_memory_utilization=MEM, trust_remote_code=True, max_num_seqs=4, enforce_eager=False,
            enable_prefix_caching=True,
            compilation_config={"mode": "none", "cudagraph_capture_sizes": [1, 2, 4, 528], "max_cudagraph_capture_size": 528})
        def gen(n): return llm.generate([short], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))[0].outputs[0].text
        shut = lambda: None
    # warmup
    gen(16)
    print(f"{'length':>7} {'best_tps':>9} {'ms_per_tok':>10}", flush=True)
    for n in [256, 512, 1024]:
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter(); gen(n); dt = time.perf_counter() - t0
            best = max(best, n / dt)
        print(f"{n:>7} {best:>9.1f} {1000/best:>10.1f}", flush=True)
    shut(); print(f"[decode] DONE {args.engine}", flush=True)


if __name__ == "__main__":
    main()
