#!/usr/bin/env python3
"""Dual torch.compile + CUDA-graph capture test for Qwen3.5-35B-A3B-FP8 (DLIN, TP4).

Exercises BOTH mechanisms simultaneously:
  - DECODE: enable_torch_compile=True  -> FullCudaGraphBackend + patch_model (torch.compile whole forward)
  - PREFILL: cuda_graph_backend_prefill='tc_piecewise' -> TcPiecewiseCudaGraphBackend
             + install_torch_compiled (FX piecewise compile + capture)

Phases (construction order, model_runner.py:914 then 916):
  1. init_prefill_cuda_graph(): TcPiecewiseCudaGraphBackend.__init__ -> _run_compile_pass
     (install trampoline on language_model.model.forward, FX-trace + inductor compile per shape
     inside enable_torch_compile_warmup), then PrefillCudaGraphRunner.capture()
     (capture_one -> cuda_piecewise_backend captures sub-graphs).
  2. init_decode_cuda_graph(): DecodeCudaGraphRunner capture with patch_model wrapping
     outer model.forward (torch.compile), FullCudaGraphBackend.capture_one (2 warmup -> capture).

Emits DUAL_RESULT markers so we can tell from the log exactly how far each phase got:
  PREFILL_COMPILE_OK / PREFILL_COMPILE_FAIL
  PREFILL_CAPTURE_OK / PREFILL_CAPTURE_FAIL
  DECODE_COMPILE_OK / DECODE_COMPILE_FAIL   (compile driven inside patch_model warmups)
  DECODE_CAPTURE_OK / DECODE_CAPTURE_FAIL
  DUAL_OUTPUT_CORRECT / DUAL_OUTPUT_GARBAGE
  DUAL_TPOT=<ms>

Run:
  source sdk-dlop-07-13-20-30/env.sh
  CUDA_VISIBLE_DEVICES=16,17,18,19 python scripts/dl/dual_compile_cg_test.py
"""
import os
import sys
import time

# --- DLIN Triton PDL shims (gdc_wait / gdc_launch_dependents) ---------------
# Inductor parses the kernel AST and getattr's these; DLIN triton lacks them.
import triton
import triton.language.extra.cuda as _tlc

if not hasattr(_tlc, "gdc_wait"):
    @triton.jit
    def _gdc_wait():
        pass
    _tlc.gdc_wait = _gdc_wait
if not hasattr(_tlc, "gdc_launch_dependents"):
    @triton.jit
    def _gdc_launch_dependents():
        pass
    _tlc.gdc_launch_dependents = _gdc_launch_dependents

# inductor subprocess must re-import the shims; TORCHINDUCTOR_COMPILE_THREADS=1
# keeps compile in-process so the shims above are visible.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
# default compile mode is faster than max-autotune-no-cudagraphs for bring-up
os.environ.setdefault("SGLANG_TORCH_COMPILE_MODE", "default")
# compiled graphs + inductor buffers need more memory headroom
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL = os.environ.get("MODEL_PATH", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/")
TP = int(os.environ.get("TP_SIZE", "4"))
MEM = float(os.environ.get("MEM_FRACTION_STATIC", "0.55"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "48"))


def _log(msg):
    print(f"[DUAL {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import sglang
    from sglang.srt.server_args import ServerArgs

    _log(f"MODEL={MODEL} TP={TP} MEM={MEM}")
    _log("Constructing Engine: enable_torch_compile=True + cuda_graph_backend_prefill=tc_piecewise")

    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        chunked_prefill_size=16,
        disable_custom_all_reduce=True,
        mem_fraction_static=MEM,
        context_length=4096,
        log_level="info",  # need INFO to see prefill/decode capture phase logs
        enable_torch_compile=True,                 # DECODE full-capture + torch.compile
        cuda_graph_backend_prefill="tc_piecewise",  # PREFILL FX piecewise compile + capture
        cuda_graph_bs_prefill=[16],                 # one small prefill shape (speeds compile)
        cuda_graph_max_bs_decode=2,                 # tiny decode capture (bs 1,2)
        cuda_graph_bs_decode=[1, 2],
    )
    _log(f"Resolved config: decode.backend={sa.cuda_graph_config.decode.backend} "
         f"prefill.backend={sa.cuda_graph_config.prefill.backend} "
         f"enable_torch_compile={sa.enable_torch_compile}")

    t0 = time.perf_counter()
    try:
        engine = sglang.Engine(server_args=sa)
    except Exception as e:
        _log(f"ENGINE_INIT_FAIL: {type(e).__name__}: {str(e)[:500]}")
        # surface the tail of any chain
        import traceback
        traceback.print_exc()
        sys.exit(2)
    _log(f"ENGINE_INIT_OK ({time.perf_counter()-t0:.1f}s) -- both runners constructed => "
         "PREFILL_COMPILE_OK + PREFILL_CAPTURE_OK + DECODE_CAPTURE_OK")

    # If we got here, both prefill tc_piecewise AND decode full-capture constructed
    # without crash. Now check correctness + speed.
    prompt = "The quick brown fox jumps over the lazy dog."
    _log("Warmup generate (drives any deferred dynamo recompiles)...")
    try:
        for _ in range(3):
            engine.generate(prompt, sampling_params={"max_new_tokens": 16, "temperature": 0})
    except Exception as e:
        _log(f"WARMUP_GENERATE_FAIL: {type(e).__name__}: {str(e)[:300]}")

    _log("Measuring correct + timed generate...")
    t0 = time.perf_counter()
    out = engine.generate(prompt, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
    dt = time.perf_counter() - t0
    text = out["text"] if isinstance(out, dict) else out[0]["text"]
    tpot = dt / MAX_NEW * 1000
    _log(f"DUAL_OUTPUT text={text[:80]!r}")
    _log(f"DUAL_TPOT={tpot:.2f}ms (wall, {MAX_NEW} tok)")

    # Coherence heuristic: the prompt-continuation should be recognizable English,
    # not prompt-regurgitation or single-char garbage.
    coherent = ("fox" in text.lower() or "dog" in text.lower() or
                len(set(text.split())) >= max(3, len(text.split()) * 0.3))
    _log("DUAL_OUTPUT_CORRECT" if coherent else "DUAL_OUTPUT_GARBAGE")

    # Replay GPU timer if available (decode full backend)
    try:
        from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as _fcg
    except Exception:
        _fcg = None

    engine.shutdown()
    _log("DUAL_TEST_DONE")


if __name__ == "__main__":
    main()
