#!/usr/bin/env python
# DLIN DFLASH-on-Qwen3.5-35B-A3B empirical probe.
# Goal: capture the REAL failure when pointing DFLASH at the hybrid MoE target
# with the only available trained draft (Qwen3-8B-DFlash-b16).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=16,17,18,19 python scripts/dl/dflash_repro.py
import os
import sys
import traceback

TARGET = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
# Only trained DFlash draft checkpoint available on this host:
DRAFT = "/mars/aebox/LLM/model/Qwen3-8B-DFlash-b16"
BLOCK_SIZE = 16

print(f"[dflash-repro] target={TARGET}", flush=True)
print(f"[dflash-repro] draft ={DRAFT}", flush=True)
print(f"[dflash-repro] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

# --- config incompatibility proof (fast, no GPU) ---
import json

t_cfg = json.load(open(os.path.join(TARGET, "config.json")))["text_config"]
d_cfg = json.load(open(os.path.join(DRAFT, "config.json")))

print("\n=== CONFIG COMPARISON (target text_config vs draft) ===")
for k in ["hidden_size", "num_hidden_layers", "vocab_size", "model_type"]:
    tv = t_cfg.get(k)
    dv = d_cfg.get(k)
    flag = "" if tv == dv else "   <-- MISMATCH"
    print(f"  {k:18s} target={tv!r:>12}  draft={dv!r:>12}{flag}")
d_target_layers = d_cfg.get("num_target_layers")
print(f"  target real num_hidden_layers = {t_cfg.get('num_hidden_layers')}  "
      f"(draft expects num_target_layers={d_target_layers})")
print(f"  draft target_layer_ids        = {d_cfg.get('dflash_config', {}).get('target_layer_ids')}")
print(f"  draft layer_types             = {d_cfg.get('layer_types')}")
print(f"  target layer_types sample     = {t_cfg.get('layer_types')[:6]} ...")
print(f"  target is hybrid MoE (linear_attention+full_attention, num_experts={t_cfg.get('num_experts')})")

mismatch = (
    t_cfg.get("hidden_size") != d_cfg.get("hidden_size")
    or t_cfg.get("vocab_size") != d_cfg.get("vocab_size")
)
print(f"\n=> hidden/vocab mismatch = {mismatch}  (DFlash draft feeds target embedding directly "
      "into draft layer 0 and shares the target lm_head)")

# --- attempt the actual Engine build (slow: 3-5 min target load) ---
print("\n=== ATTEMPTING sglang.Engine WITH DFLASH ===", flush=True)
try:
    import sglang as sgl

    # num_steps=1, eagle_topk=1, pp_size=1 per DFLASH contract.
    engine_kwargs = dict(
        model_path=TARGET,
        tp_size=4,
        dtype="bfloat16",
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=0.7,  # leave room for draft KV pool
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        # DFLASH spec args:
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT,
        speculative_dflash_block_size=BLOCK_SIZE,
    )
    print(f"[dflash-repro] Engine kwargs: { {k: v for k, v in engine_kwargs.items()} }", flush=True)
    engine = sgl.Engine(**engine_kwargs)
    print("[dflash-repro] ENGINE BUILT OK — attempting a greedy generate", flush=True)
    try:
        out = engine.generate(
            {"input_ids": None, "text": "The quick brown fox jumps over the lazy dog.",
             "sampling_params": {"max_new_tokens": 8, "temperature": 0}},
        )
        print(f"[dflash-repro] DFLASH generate output: {out}", flush=True)
    finally:
        engine.shutdown()
    print("[dflash-repro] DFLASH RUN COMPLETED (no exception)", flush=True)
except SystemExit:
    raise
except Exception as e:
    print("\n=== DFLASH LOAD/RUN FAILED ===", flush=True)
    print(f"Exception type: {type(e).__name__}", flush=True)
    print(f"Exception msg : {e}", flush=True)
    print("\n--- full traceback ---", flush=True)
    traceback.print_exc()
    sys.exit(2)
