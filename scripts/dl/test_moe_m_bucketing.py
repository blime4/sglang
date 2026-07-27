#!/usr/bin/env python3
"""Is the fused-MoE dlcc JIT keyed on EXACT M, or bucketed?
Warm M=512 (one prefill), then prefill lengths that produce a PARTIAL last chunk
(576=512+64, 640=512+128, 768=512+256, 1024=512+512). If the partial-chunk M JITs
separately, those calls are slow (~20-85s); if bucketed to 512, fast & ~flat.
"""
import os, time
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "2048")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")


def main():
    import sglang as sgl
    MODEL = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8"
    TP = int(os.environ.get("TP_SIZE", "4"))
    e = sgl.Engine(model_path=MODEL, tp_size=TP, dtype="bfloat16", context_length=4096,
        mem_fraction_static=0.55, max_running_requests=4, disable_cuda_graph=False,
        cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
        disable_custom_all_reduce=True, trust_remote_code=True, chunked_prefill_size=512)
    print("[bucket] warming M=512 (one prefill, untimed)...", flush=True)
    t0 = time.perf_counter(); e.generate("x " * 512, {"max_new_tokens": 1, "temperature": 0})
    print(f"[bucket] warm M=512 took {time.perf_counter()-t0:.1f}s", flush=True)

    def prefs(n):
        p = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * (n // 13 + 1))[:n*5]
        t0 = time.perf_counter(); e.generate(p, {"max_new_tokens": 1, "temperature": 0})
        return time.perf_counter() - t0

    print(f"\n{'length':>7} {'full':>5} {'partial_M':>9} {'time_ms':>9} {'verdict':>12}", flush=True)
    for n in [512, 576, 640, 768, 1024]:
        full = n // 512; partial = n % 512
        dt = prefs(n)
        verdict = "WARM(fast)" if dt < 1.5 else "JIT(slow)"
        print(f"{n:>7} {full:>5} {partial:>9} {dt*1000:>9.0f} {verdict:>12}", flush=True)
    e.shutdown(); print("[bucket] DONE", flush=True)


if __name__ == "__main__":
    main()
