#!/usr/bin/env python3
"""Eager baseline for the dual compile+CG comparison (same model/TP/GPUs/bs).

Matches dual_compile_cg_test.py EXACTLY except:
  - enable_torch_compile=False (decode = FullCudaGraphBackend, no torch.compile)
  - cuda_graph_backend_prefill='disabled' (prefill eager, no tc_piecewise)
This isolates the effect of compile+CG vs plain CG (eager graph replay).
"""
import os, sys, time
import triton, triton.language.extra.cuda as _tlc
if not hasattr(_tlc, "gdc_wait"):
    @triton.jit
    def _gdc_wait(): pass
    _tlc.gdc_wait = _gdc_wait
if not hasattr(_tlc, "gdc_launch_dependents"):
    @triton.jit
    def _gdc_launch_dependents(): pass
    _tlc.gdc_launch_dependents = _gdc_launch_dependents

os.environ.setdefault("SGLANG_DL_TIME_REPLAY", "1")
MODEL = os.environ.get("MODEL_PATH", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/")
TP = int(os.environ.get("TP_SIZE", "4"))
MEM = float(os.environ.get("MEM_FRACTION_STATIC", "0.55"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "48"))

def _log(m): print(f"[EAGER {time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    import sglang
    from sglang.srt.server_args import ServerArgs
    _log("Eager baseline: enable_torch_compile=False, prefill=disabled (plain CG)")
    sa = ServerArgs(
        model_path=MODEL, dtype="bfloat16", tp_size=TP,
        attention_backend="fa3", page_size=16, chunked_prefill_size=16,
        disable_custom_all_reduce=True, mem_fraction_static=MEM,
        context_length=4096, log_level="info",
        enable_torch_compile=False,
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=2, cuda_graph_bs_decode=[1, 2],
    )
    _log(f"Resolved: decode={sa.cuda_graph_config.decode.backend} prefill={sa.cuda_graph_config.prefill.backend}")
    t0 = time.perf_counter()
    engine = sglang.Engine(server_args=sa)
    _log(f"ENGINE_INIT_OK ({time.perf_counter()-t0:.1f}s)")
    p = "The quick brown fox jumps over the lazy dog."
    for _ in range(3):
        engine.generate(p, sampling_params={"max_new_tokens": 16, "temperature": 0})
    t0 = time.perf_counter()
    out = engine.generate(p, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
    dt = time.perf_counter() - t0
    text = out["text"] if isinstance(out, dict) else out[0]["text"]
    _log(f"EAGER_OUTPUT text={text[:80]!r}")
    _log(f"EAGER_TPOT={dt/MAX_NEW*1000:.2f}ms (wall, {MAX_NEW} tok)")
    engine.shutdown()
    _log("EAGER_TEST_DONE")

if __name__ == "__main__":
    main()
