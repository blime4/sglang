#!/usr/bin/env python3
"""Systematic benchmark: multiple prompt lengths × generation lengths × configs.
Run 3 trials each, report median."""
import os, time, sglang, statistics

MODEL = os.environ["MODEL_PATH"]
TP = int(os.environ.get("TP", "2"))
FUSED = os.environ.get("SGLANG_DL_MOE_FUSED", "1")
CG = os.environ.get("CG", "1") == "1"

PROMPTS = {
    "short":  "The capital of France is",
    "medium": "Write a detailed essay about the history of computing and artificial intelligence. The essay should cover the early days of mechanical computers,",
    "long":   "Please write a comprehensive technical report about the architecture of modern large language models. Start with an introduction to transformer architectures, then discuss attention mechanisms in detail, including multi-head attention, grouped-query attention, and multi-query attention. Cover the evolution from the original Transformer paper through BERT, GPT series, and modern models. Include discussions of training methodologies like pre-training, fine-tuning, RLHF, and DPO. Discuss scaling laws and how they inform model design decisions. Cover inference optimizations like quantization, speculative decoding, and key-value caching. The report should be approximately 2000 words and suitable for a graduate-level audience.",
}

GEN_LENGTHS = [50, 200, 500]
TRIALS = 3

def main():
    e = sglang.Engine(
        model_path=MODEL, dtype="bfloat16", tp_size=TP,
        attention_backend="fa3", page_size=16, mem_fraction_static=0.85,
        disable_cuda_graph=not CG, cuda_graph_max_bs_decode=2,
        context_length=4096)
    # warmup (trigger all JIT)
    e.generate("hello world test warmup", sampling_params={"max_new_tokens": 8, "temperature": 0})

    print(f"{'prompt':>8} {'gen_len':>8} {'trial':>6} {'tok/s':>8} {'total_s':>8}  output_check")
    results = {}
    for plabel, prompt in PROMPTS.items():
        for gen_len in GEN_LENGTHS:
            speeds = []
            for trial in range(TRIALS):
                t0 = time.time()
                r = e.generate(prompt, sampling_params={"max_new_tokens": gen_len, "temperature": 0})
                dt = time.time() - t0
                tps = gen_len / dt
                speeds.append(tps)
                check = r["text"][:20].replace("\n", "\\n")
                print(f"{plabel:>8} {gen_len:>8} {trial:>6} {tps:>8.2f} {dt:>8.1f}  {check!r}")
            med = statistics.median(speeds)
            results[(plabel, gen_len)] = med
            print(f"{plabel:>8} {gen_len:>8} {'median':>6} {med:>8.2f}")
            print()

    print("\n=== SUMMARY (median tok/s) ===")
    print(f"{'prompt':>8} {'gen_len':>8} {'tok/s':>8}")
    for (plabel, gen_len), med in sorted(results.items()):
        print(f"{plabel:>8} {gen_len:>8} {med:>8.2f}")
    e.shutdown()

if __name__ == "__main__":
    main()
