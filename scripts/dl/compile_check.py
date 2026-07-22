#!/usr/bin/env python3
"""Fast torch.compile diagnostic for DLIN sglang (Qwen3.5-35B-A3B-FP8, TP4).

Mirrors /tmp/sg_compile.py's compile config but WITHOUT TORCH_LOGS=graph_breaks
(that verbosity made a single run take 20+ min). Instead it prints:
  - COMPILE_TPOT=<ms>   (best-of-3 steady-state decode)
  - GRAPH_BREAKS=<n>    (from torch._dynamo.utils.counters["stats"])
  - a compact reason histogram

Run:
  source $SDK_DIR/env.sh
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/dl/compile_check.py
  # optional: VERBOSE_GB=1 to also dump full TORCH_LOGS=graph_breaks (slow)
"""
import os

# torch.compile + CUDA-graph capture uses more GPU memory than eager (inductor
# buffers + captured-graph private pools). Defragment + leave headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
MEM_FRAC = float(os.environ.get("MEM_FRACTION_STATIC", "0.50"))

MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TP = int(os.environ.get("TP_SIZE", "4"))


def main():
    if os.environ.get("VERBOSE_GB") == "1":
        os.environ["TORCH_LOGS"] = "graph_breaks"

    import torch
    import sglang
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(
        model_path="/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/",
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=MEM_FRAC,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=2,
        context_length=4096,
        disable_custom_all_reduce=True,
        log_level="error",
        enable_torch_compile=True,
    )
    try:
        engine = sglang.Engine(server_args=sa)
        p = "The quick brown fox jumps over the lazy dog."
        for _ in range(5):  # warmup (drives dynamo trace + inductor compile)
            engine.generate(p, sampling_params={"max_new_tokens": 16, "temperature": 0})
        out0 = engine.generate(p, sampling_params={"max_new_tokens": 32, "temperature": 0})
        best = 9999.0
        import time
        for _ in range(3):
            t0 = time.perf_counter()
            engine.generate(p, sampling_params={"max_new_tokens": MAX_NEW, "temperature": 0})
            best = min(best, (time.perf_counter() - t0) / MAX_NEW * 1000)
        print(f"COMPILE_TPOT={best:.2f}ms", flush=True)
        print(f"SAMPLE_OUTPUT={out0['text'][:60]!r}", flush=True)
        engine.shutdown()
    except Exception as e:
        print(f"COMPILE_FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        raise

    # ---- dynamo break summary (cheap; no verbose logging) ----
    try:
        from torch._dynamo.utils import counters

        stats = counters.get("stats", {})
        gb = stats.get("unique_graph_breaks", stats.get("graph_breaks", 0))
        print(f"GRAPH_BREAKS={gb}", flush=True)
        # reason histogram if available
        br = counters.get("break_reasons", {})
        if br:
            print("BREAK_REASONS:")
            for reason, n in sorted(br.items(), key=lambda x: -x[1])[:10]:
                print(f"  {n:>4}  {reason[:90]}")
        else:
            print("(no per-reason counter; set VERBOSE_GB=1 for full graph_break log)")
    except Exception as e:
        print(f"(counter dump failed: {e})", flush=True)


if __name__ == "__main__":
    main()
