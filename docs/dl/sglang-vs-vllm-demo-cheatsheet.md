# sglang vs vLLM on DLIN — Demo Cheat Sheet

> One page. Qwen3.6-35B-A3B-FP8, TP4 (KS38 ×4), FP8, same GPUs, fresh process each,
> best-of-N. Verified + fairness-audited 2026-07-25/26 (runs r015/r016). Full detail:
> [`sglang-vs-vllm-rigor-analysis.md`](sglang-vs-vllm-rigor-analysis.md),
> [`sglang-vs-vllm-fairness-defense.md`](sglang-vs-vllm-fairness-defense.md).

## Headline

**sglang beats vLLM 6–16× on every KV-reuse workload; vLLM wins raw decode + raw
prefill (shown openly). The comparison is fair and reproducible.** Wins are specific
to this **hybrid-Mamba** model class (where vLLM MRV2 cannot enable prefix caching)
and to **prefix-sharing / multi-turn / RAG / multi-tenant / concurrent** workloads.

## The numbers (r015/r016)

| SC | workload | sglang | vLLM-MRV2 | winner |
|----|----------|--------|-----------|--------|
| SC1 | prefix sharing (warm latency) | 2022 ms | 12887 ms | **sglang 6.4×** |
| SC2 | multi-turn (turn-5 latency) | 2258 ms | 15773 ms | **sglang 7.0×** |
| SC3 | concurrent batch (throughput) | 19.8 tok/s | 2.6 | **sglang 7.6×** |
| SC5 | multi-user fork (radix tree) | 13.1 s | 108.6 s | **sglang 8.3×** |
| SC7 | long-RAG (~2K doc × 8 queries) | 13.1 tok/s | 0.8 | **sglang 16.4×** |
| SC8 | repeated best-of-N (RLHF loop) | 22.2 tok/s | 3.7 | **sglang 6.0×** |
| SC8b | cold single best-of-N (rigor) | 5.7 tok/s | 2.8 | **sglang 2.0×** |
| SC10 | shared system-prompt (12 tenants) | 13.3 tok/s | 1.9 | **sglang 7.0×** |
| SC11 | **online concurrency** (unequal, batch) | 17.4 tok/s | 1.9 | **sglang 9.2×** |
| SC6 | raw-prefill parity (unique, no cache) | 49 tok/s | **76** | **vLLM 1.55×** |
| SC9 | pure long decode (short prompt) | 30.6 tok/s | **39.5** | **vLLM 1.29×** |

SC1 cold→warm "speedup" (10.3×) is JIT-contaminated — cite the **warm latency** (6.4×),
not the speedup. SC9's gap is ~1.13× at sglang's tuned optimum.

## Reproduce (one command)

```bash
./run_sglang.sh -M qwen35-35b compare --scenarios SC1,SC2,SC3,SC5,SC6,SC7,SC8,SC9,SC10,SC11
# per-commit history: ./run_sglang.sh compare --history
```

## "Is this fair?" — one-line rebuttals (full: fairness-defense.md)

- **"You disabled vLLM prefix caching (APC)"** → `apc_failure_probe.py` prints the proof
  in 5 s: vLLM **MRV2** (its only *stable* DLIN runner) **cannot enable APC** on this
  hybrid-Mamba model — `mamba_cache_mode='align'` unsupported (`vllm/config/vllm.py:2030`).
  MRV1 can (eager) but is unstable on DLIN. APC-off is **forced**, not chosen. (Dense
  control: on Qwen3-1.7B, MRV2+APC works → the limit is hybrid-Mamba-specific.)
- **"Rigged toward sglang"** → sglang **loses openly** on SC6 (raw prefill, vLLM 1.55×)
  and SC9 (decode, vLLM 1.29×). Wins are KV-reuse only.
- **"Outputs differ"** → greedy **identical** on short decode; diverge only on long
  prefill (cross-engine FP8 drift, both coherent). Speed metric is token-count-controlled.
- **Scope** → wins are **hybrid-Mamba + KV-reuse**; on dense models vLLM caches too and
  the gap shrinks. Stated honestly.

## What's behind the wins

sglang **RadixAttention** keeps the shared KV and only extends the small per-request
suffix; vLLM MRV2 re-prefills the whole shared prefix every request (APC-off). The
longer/more-shared the prefix, the bigger the gap (SC7's ~2K doc = 16.4×). Online
concurrency (SC11) was unlocked this cycle by the FA2 unequal-length-prefill fix.
