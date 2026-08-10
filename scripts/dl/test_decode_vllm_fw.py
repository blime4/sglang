#!/usr/bin/env python3
"""vLLM decode with FULL warmup (gen 512) — fair comparison vs sglang's 43.2."""
import os, time
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


def main():
    from vllm import LLM, SamplingParams
    M = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    llm = LLM(model=M, tensor_parallel_size=4, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.55, trust_remote_code=True, max_num_seqs=4, enforce_eager=False,
        enable_prefix_caching=True,
        compilation_config={"mode": "none", "cudagraph_capture_sizes": [1, 2, 4, 528], "max_cudagraph_capture_size": 528})
    sp = SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True)
    P = "Write a detailed step-by-step plan for deploying a model serving system."
    llm.generate([P], sp)  # FULL warmup (512 tokens)
    reps = []
    for _ in range(5):
        t0 = time.perf_counter(); llm.generate([P], sp); reps.append(512 / (time.perf_counter() - t0))
    import statistics
    print(f"[decode-vllm] 5 reps (tok/s): {[round(r,1) for r in reps]} "
          f"| mean={statistics.mean(reps):.1f} min={min(reps):.1f} max={max(reps):.1f}", flush=True)
    print("[decode-vllm] DONE", flush=True)


if __name__ == "__main__":
    main()
