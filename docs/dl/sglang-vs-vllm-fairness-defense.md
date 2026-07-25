# sglang vs vLLM Compare — Fairness Defense (Demo Rebuttal Sheet)

> Companion to [`sglang-vs-vllm-rigor-analysis.md`](sglang-vs-vllm-rigor-analysis.md).
> One-line answer to every "is this comparison fair?" critique, with the reproducible
> proof. Model: Qwen3.5/3.6-35B-A3B-FP8, TP4, DLIN. Verified 2026-07-25.

## The big one — "You disabled vLLM prefix caching (APC), so of course sglang wins"

**APC cannot be enabled in vLLM's stable config on this model — it's a hard limit, not a choice.** Proof: `scripts/dl/apc_failure_probe.py` (run it: `CUDA_VISIBLE_DEVICES=… .venv/bin/python scripts/dl/apc_failure_probe.py`), which tests **both** vLLM runners explicitly:

- **MRV2** (the compare's stable DLIN runner) + `enable_prefix_caching=True` → **FAILS**:
  - under CG: `block_size(528) > max_num_batched_tokens(4)` (DLIN CG clamps the batch);
  - under eager: `AssertionError: Model Runner V2 has not yet supported mamba_cache_mode='align'` (`vllm/config/vllm.py:2030`).
  - APC forces `mamba_cache_mode='align'` for hybrid-Mamba models, which **MRV2 does not support**.
- **MRV1** + APC + eager → works (correct output), **but MRV1 is unstable on DLIN** (crashes `ConstraintViolationError`/assert — the compare skips it by default).

⇒ vLLM's **only stable working config** on this model is **APC-OFF**. We compare each engine's best *stable* config — the only fair comparison available. **One-line demo rebuttal: "run `apc_failure_probe.py` — MRV2 can't enable APC on this model; here's the assertion."**

(Why this matters: hybrid-Mamba models are the cutting edge — Qwen3.5/3.6, Jamba, Zamba. On these, sglang RadixAttention works natively *and* with CG; vLLM MRV2 can do neither-prefix-cache. That's a structural model-class edge, not a tuning gap. On a **dense** model where MRV2+APC works, the gap shrinks — see the dense control below.)

## "sglang wins every scenario — it's rigged toward sglang"

**No — sglang loses where it has no structural edge.** The compare deliberately includes scenarios vLLM wins:

- **SC6 raw-prefill parity** (unique prompts, no cache): vLLM **76 tok/s** vs sglang **49** → **vLLM 1.55× faster** at raw prefill. Proves the prefix-scenario wins are *caching*, not raw speed.
- **SC9 pure decode** (short prompt, no shared structure): vLLM **39.6** vs sglang **30.5-35 tok/s** → **vLLM 1.13-1.29×**. vLLM's decode-IPC edge.

So the picture is balanced: sglang wins KV-reuse workloads (SC1-3,5,7,8,10,11, 2-16×); vLLM wins raw prefill (SC6) + raw decode (SC9). Each engine's strength is shown.

## "The engine configs aren't equally tuned"

Same model, TP4, FP8, mem 0.55, CG on (capture [1,2,4]), max_seqs=4, same prompts, temp 0 (SC8=0.7), best-of-N, fresh process each. See each `{tag}.log` `COMMAND`+`CONFIG` header (logged by `run_sglang.sh compare`). The MoE/attention kernels are literally the same `_dl_C.so` for both engines. [rigor §1b]

## "The outputs differ, so it's not a fair speed comparison"

Greedy outputs: **identical** on short-prefill pure decode (SC9, 48/48 tokens); diverge only on long prefills (cross-engine FP8 drift, greedy-amplified — both coherent, neither wrong) [rigor §1d, `scripts/dl/greedy_agreement.py`]. The speed metric is **token-count-controlled** (`ignore_eos=True` on SC1/2/3) so divergence can't skew throughput. Speed = tokens/sec is token-identity-independent.

## "Cold-start / JIT contamination favors sglang"

Both engines: 3x warmup + best-of-N, fresh process each config. SC1's cold→warm *speedup* is JIT-contaminated (reported as a range, not a clean metric — F5); all other metrics are steady-state best-of-N. [rigor §3 SC1, §1c]

## Honest scope of the claims

- Wins are **specific to hybrid-Mamba models** (where vLLM MRV2 can't enable APC). On a dense model, vLLM MRV2+APC works and the prefix-reuse gaps shrink.
- Wins are **KV-reuse workloads** (prefix sharing, multi-turn, RAG, multi-tenant, online concurrency).
- sglang **loses** raw decode (SC9) and raw prefill (SC6) — stated openly.

## Dense-model control (APC works where the model allows it) — CONFIRMED

To show this isn't "cherry-picked a model where vLLM is broken": ran `apc_failure_probe.py` on **dense Qwen3-1.7B** (TP1) — **MRV2+APC WORKS** (correct output p1→192, p2→4; MRV1+APC also works). So APC **is** enableable on MRV2 for a dense model. The APC limitation is **hybrid-Mamba-specific** (MRV2 can't do `mamba_cache_mode='align'`, which APC forces for Mamba layers) — not a general vLLM cripple. This scopes the claim honestly: sglang's prefix-reuse edge holds on **hybrid-Mamba** models (Qwen3.5/3.6, Jamba, Zamba); on dense models vLLM MRV2+APC works and the gap would shrink.

---
**Bottom line for a demo:** the comparison is fair — same hardware/model/precision, each engine's best *stable* config, sglang loses openly where it has no edge, and the one critique that could sink it ("enable APC") is defused by a 5-second reproducible proof that MRV2 cannot enable APC on this model class.
