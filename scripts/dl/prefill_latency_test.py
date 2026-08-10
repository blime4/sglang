#!/usr/bin/env python3
"""Prefill latency: dual compile+CG (prefill tc_piecewise) vs eager, on DLIN.

Measures pure prefill cost (long prompt, max_new_tokens=1) so the timing is
dominated by the prefill forward, isolating whether prefill tc_piecewise is
beneficial / neutral / harmful vs eager.

Env: COMPILE=1 (default) enables torch.compile (dual-capture default on DLIN
after commit a838ae1f56). COMPILE=0 = eager baseline.
    source sdk-dlop-07-13-20-30/env.sh
    CUDA_VISIBLE_DEVICES=16,17,18,19 COMPILE=1 python scripts/dl/prefill_latency_test.py
"""
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
COMPILE = os.environ.get("COMPILE", "1") == "1"
PROMPT_TOKS = int(os.environ.get("PROMPT_TOKENS", "1024"))


def main():
    import sglang
    from sglang.srt.server_args import ServerArgs

    # serving-recipe env
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
    if COMPILE:
        os.environ.setdefault("SGLANG_TORCH_COMPILE_MODE", "default")

    sa = ServerArgs(
        model_path="/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/",
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
        enable_torch_compile=COMPILE,  # dual-capture default-on on DLIN when True
    )
    print(
        f"COMPILE={COMPILE} prefill_backend={sa.cuda_graph_config.prefill.backend} "
        f"decode_backend={sa.cuda_graph_config.decode.backend}",
        flush=True,
    )
    engine = sglang.Engine(server_args=sa)
    # long prompt (~1024 toks): repeat a sentence
    sentence = "The quick brown fox jumps over the lazy dog. "
    prompt = sentence * (PROMPT_TOKS // 9)

    # warmup (drive compile + CG capture)
    for _ in range(3):
        engine.generate(prompt, sampling_params={"max_new_tokens": 1, "temperature": 0})

    # measure prefill: generate(max_new_tokens=1) time ≈ prefill time
    best = 9999.0
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = engine.generate(prompt, sampling_params={"max_new_tokens": 1, "temperature": 0})
        dt = time.perf_counter() - t0
        times.append(dt)
        best = min(best, dt)
    engine.shutdown()
    times.sort()
    med = times[len(times) // 2]
    print(f"PREFILL_RESULT compile={COMPILE} prompt_toks~{PROMPT_TOKS} "
          f"best={best*1000:.1f}ms median={med*1000:.1f}ms out={out['text'][:30]!r}",
          flush=True)


if __name__ == "__main__":
    main()
