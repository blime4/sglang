#!/usr/bin/env python3
# DL begin — sglang offline prefill captured under a cudaProfilerApi range for dlpti.
# Based on scripts/dl/prefill_latency_test.py (eager, COMPILE=0 = SC6 baseline).
# Captures ONLY the prefill forward (engine load + warmup NOT captured), keeping
# the dlpti trace small and the GPU-kernel breakdown meaningful.
#
# Run:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 dlpti_tools capture --capture-range cudaProfilerApi \
#     --activity-mask cmd,cu --data-file /tmp/prefill_dlpti.db \
#     -- python scripts/dl/profile_prefill_dlpti.py
#   dlpti_tools export --format perfetto-json /tmp/prefill_dlpti.db
import ctypes
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    import sglang
    from sglang.srt.server_args import ServerArgs

    # serving-recipe env (same as prefill_latency_test.py)
    for k, v in {
        "SGLANG_DL_GDN_DLIN": "1",
        "SGLANG_DL_MOE_FUSED": "1",
        "SGLANG_DL_MOE_FUSED_MAX_M": "16",
        "SGLANG_DL_FP8_Q2": "1",
        "DLEOL_CACHE_SIZE": "1024",
        "DLEOL_FLA_ENABLE_PINGPONG": "1",
        "DLEOL_FLA_UNROLL_COUNT": "8",
    }.items():
        os.environ.setdefault(k, v)

    sa = ServerArgs(
        model_path="/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8",
        dtype="bfloat16",
        tp_size=4,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.55,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=2,
        context_length=4096,
        disable_custom_all_reduce=True,
        log_level="error",
    )
    print(
        f"prefill_backend={sa.cuda_graph_config.prefill.backend} "
        f"decode_backend={sa.cuda_graph_config.decode.backend}",
        flush=True,
    )
    engine = sglang.Engine(server_args=sa)

    # ~1K-token prompt (SC6 raw-prefill shape ~1337 tok/prompt)
    prompt = "The quick brown fox jumps over the lazy dog. " * 114

    # warmup (drive JIT + CG capture) — NOT under the capture range
    for _ in range(3):
        engine.generate(prompt, sampling_params={"max_new_tokens": 1, "temperature": 0})

    # capture ONLY the prefill forward under cudaProfilerApi range
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart()
    engine.generate(prompt, sampling_params={"max_new_tokens": 1, "temperature": 0})
    cudart.cudaProfilerStop()

    engine.shutdown()
    print("prefill capture done", flush=True)


if __name__ == "__main__":
    main()
# DL end
