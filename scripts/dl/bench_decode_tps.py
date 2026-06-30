#!/usr/bin/env python3
# Empirical Qwen3-1.7B decode tok/s on DLIN (torch.profiler crashes on DLIN, so
# measure end-to-end generation throughput directly). Decode-dominated: short
# prompt + many new tokens, tok/s ~= new_tokens / wall_clock (prefill amortizes).
import os, time
import sglang

MODEL = os.environ.get("MODEL_PATH", "/opt/dataset/Qwen3-1.7B")
BACKEND = os.environ.get("ATTN_BACKEND", "fa3")
NEW = int(os.environ.get("MAX_NEW_TOKENS", "64"))
RUNS = int(os.environ.get("RUNS", "3"))
CG = os.environ.get("CUDA_GRAPH", "0") == "1"
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.88"))
CG_MAX_BS = int(os.environ.get("CG_MAX_BS", "0"))   # cuda_graph_max_bs_decode (0=default)
CTX = int(os.environ.get("CONTEXT_LEN", "0"))        # context_length override (0=model default)
CG_BACKEND = os.environ.get("CG_BACKEND", "")        # cuda_graph_backend_decode (""=default full)


def main():
    kw = dict(
        model_path=MODEL, page_size=16, dtype="bfloat16",
        attention_backend=BACKEND, disable_cuda_graph=not CG,
        mem_fraction_static=MEM_FRAC,
    )
    if CG and CG_MAX_BS:
        kw["cuda_graph_max_bs_decode"] = CG_MAX_BS
    if CG and CG_BACKEND:
        kw["cuda_graph_backend_decode"] = CG_BACKEND
    if CTX:
        kw["context_length"] = CTX
    engine = sglang.Engine(**kw)
    prompt = "The capital of France is"
    # warmup (JIT compile + cuda-graph capture if enabled)
    engine.generate([prompt], sampling_params={"max_new_tokens": 8})
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}]")
    print(f"{'run':>4} {'tokens':>7} {'wall_s':>8} {'tok/s':>8}")
    for r in range(RUNS):
        t0 = time.time()
        out = engine.generate([prompt], sampling_params={"max_new_tokens": NEW, "temperature": 0})
        dt = time.time() - t0
        text = out[0]["text"] if isinstance(out, list) else out["text"]
        print(f"{r:>4} {NEW:>7} {dt:>8.3f} {NEW/dt:>8.2f}")
    print(f"\n[cuda_graph={'ON' if CG else 'OFF'}] prompt: '{prompt}' -> '{text[:40]}'")


if __name__ == "__main__":
    main()
