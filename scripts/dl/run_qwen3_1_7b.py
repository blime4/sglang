#!/usr/bin/env python3
# Minimal end-to-end generation on DLIN GPU (KS38), driven by env vars so
# run_sglang.sh::gen can call it. Mirrors the invocation verified in commit
# fcf5fec1d2: sglang.Engine(page_size=16) -> coherent text.
#
# Env (all optional):
#   MODEL_PATH        default /opt/dataset/Qwen3-1.7B
#   ATTN_BACKEND      default fa3  (force FlashAttention -> DLIN FA2; the default
#                     'triton' backend hits tl.extra.cuda.gdc_launch_dependents,
#                     absent in DLIN triton 3.1.0)
#   USE_CUDA_GRAPH    default 0    (cuda-graph capture hits the same triton kernel;
#                     keep off for correctness tests)
#   PROMPT            default "" -> builtin demo prompts
#   MAX_NEW_TOKENS    default 16
#
# This script ASSUMES the clean DLIN runtime env is already active (run_sglang.sh
# sets it via dlin_runtime_env): LD_LIBRARY_PATH=$SDK/lib ONLY, CUDA_HOME=$SDK,
# DLI_V2=ON, venv-first PATH, CUDA_VISIBLE_DEVICES=<free gpu>.
import os

import sglang

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))
DISABLE_CG = os.environ.get("USE_CUDA_GRAPH", "0") != "1"

# Guard is MANDATORY: sglang.Engine spawn-starts scheduler subprocesses; under the
# spawn start method the child re-imports this module, so a top-level Engine
# creation would re-trigger spawning ("attempt to start a new process before the
# current process has finished its bootstrapping phase").
def main():
    engine = sglang.Engine(
        model_path=MODEL,
        page_size=16,    # DLIN FA2 verified at page_size=16 (commit fcf5fec1d2)
        dtype="bfloat16",
        attention_backend=BACKEND,
        disable_cuda_graph=DISABLE_CG,
    )

    prompts = [os.environ["PROMPT"]] if os.environ.get("PROMPT") else [
        "Hello, my name is",
        "The capital of France is",
    ]
    out = engine.generate(prompts, sampling_params={"max_new_tokens": MAX_NEW_TOKENS})

    # engine.generate returns a dict (single str) or list[dict] (list input).
    results = out if isinstance(out, list) else [out]
    for p, r in zip(prompts, results):
        text = r["text"] if isinstance(r, dict) else str(r)
        print(f"\n[PROMPT ] {p}\n[OUTPUT ] {p}{text}")


if __name__ == "__main__":
    main()
