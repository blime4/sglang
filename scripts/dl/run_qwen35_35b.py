#!/usr/bin/env python3
# Bring-up + minimal generate for Qwen3.5-35B-A3B-FP8 on sglang/DLIN.
# Goal part 1 (跑通): load + generate correct text. TP=2 (35B FP8 ≈ 37GB > 1x32GB).
# NOTE: Engine() MUST be under __main__ (sglang spawns scheduler via spawn).
import os

# DL: DLIN Triton lacks the Hopper PDL extras gdc_wait / gdc_launch_dependents. FLA
# linear-attn kernels reference them in the AST (Triton hashes even the constexpr-dead
# branch), so they must resolve + be callable. USE_GDC=False at runtime (is_arch_support_pdl
# patched) means these are never executed; empty @triton.jit no-ops satisfy hash + compile.
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

import sglang

MODEL = os.environ.get("MODEL_PATH", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP", "2"))
MEM = float(os.environ.get("MEM_FRAC", "0.82"))
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")


def main():
    kw = dict(
        model_path=MODEL,
        dtype="bfloat16",  # FP8 weights auto-detected from quantization_config
        tp_size=TP,
        attention_backend=BACKEND,
        page_size=16,
        mem_fraction_static=MEM,
        disable_cuda_graph=True,  # eager for bring-up (correctness first)
    )
    engine = sglang.Engine(**kw)
    out = engine.generate(
        ["The capital of France is"],
        sampling_params={"max_new_tokens": 16, "temperature": 0},
    )
    text = out[0]["text"] if isinstance(out, list) else out["text"]
    print("\n===== GENERATED =====")
    print(repr(text))
    print("===== DONE =====")


if __name__ == "__main__":
    main()
