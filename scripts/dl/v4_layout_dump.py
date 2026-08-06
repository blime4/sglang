#!/usr/bin/env python3
"""DL: DSV4-Flash runtime FP4 layout dump — one MoE layer, TP8, spawn-safe via
SGLANG_DL_V4_DUMP gate in fp8.py. Run:
  export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 TP_SIZE=8
  .venv/bin/python scripts/dl/v4_layout_dump.py
"""
import os, time, sys, json

for _k, _v in {
    "SGLANG_DL_FP8_Q2": "1", "SGLANG_DL_MOE_FUSED": "1", "SGLANG_DL_MOE_FUSED_MAX_M": "2048",
    "SGLANG_DL_GDN_DLIN": "1", "DLEOL_CACHE_SIZE": "1024",
    "SGLANG_DL_V4_DUMP": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(_k, _v)

MODEL = os.environ.get("MODEL_PATH", "/LocalRun/hao.dong/DeepSeek-V4-Flash")
TP = int(os.environ.get("TP_SIZE", "8"))

if __name__ == "__main__":
    import sglang as sgl
    print(f"[v4-dump] model={MODEL} tp={TP} @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    engine = sgl.Engine(
        model_path=MODEL, tp_size=TP, dtype="bfloat16",
        trust_remote_code=True, mem_fraction_static=0.90,
        disable_custom_all_reduce=True, disable_cuda_graph=True,
    )
    print(f"[v4-dump] engine up in {time.perf_counter()-t0:.0f}s", flush=True)
    out = engine.generate("The capital of France is", {"max_new_tokens": 4, "temperature": 0, "ignore_eos": True})
    txt = out["text"] if isinstance(out, dict) else str(out)
    print(f"[v4-dump] GENERATED: {txt!r}", flush=True)
    engine.shutdown()
    print("[v4-dump] DONE", flush=True)
