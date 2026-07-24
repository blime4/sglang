# sglang vs vLLM Showcase — Rigor Analysis (Are the Wins Fair?)

> Companion to [`sglang-vs-vllm-new-scenarios.md`](sglang-vs-vllm-new-scenarios.md)
> (the SC5/SC7/SC8/SC9/SC10 results) and [`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md)
> (SC1–SC4). This doc **stress-tests the rigor of each new scenario** — is each
> sglang-vs-vLLM comparison fair, or is a confound inflating the gap? — and gives
> the evidence chain ("supporting path") behind every verdict.
>
> Model: Qwen3.5/3.6-35B-A3B-FP8, TP4, DLIN. Commit `b22ee6c393` + the SC6/SC8b
> diagnostic probes added 2026-07-24.

## TL;DR — verdict per scenario

| Scenario | sglang vs vLLM | Rigor verdict |
|----------|----------------|---------------|
| **SC5** multi-user fork | sglang **8.22×** | ✅ **Rigorous** — win is RadixAttention caching (proven: not raw prefill, see SC6) |
| **SC7** long-RAG | sglang **16.25×** | ✅ **Rigorous** — win is caching |
| **SC10** shared system-prompt | sglang **7.05×** | ✅ **Rigorous** — win is caching |
| **SC8** best-of-N (same prompt reused) | sglang **5.84×** | ⚠️ **Mixed** — decomposes (via SC8b) into: cross-call prompt caching (~3.9×, RLHF-loop) × a real single-call best-of-N edge (sglang ~2×; vLLM n=4 is pathologically slow on DLIN). Both factors are real sglang wins but measure different workloads — see §3 SC8. |
| **SC9** pure decode | vLLM **1.30×** | ⚠️ **Direction rigorous, magnitude caveat** — vLLM wins decode (robust), but 1.30× is an upper bound: sglang's decode here (30.5 tok/s) is below its ~35 optimum, so the true gap is ~1.13×. |

**Bottom line:** the three big prefix-reuse wins (SC5/SC7/SC10, 7–16×) are real
and correctly attributed — SC6 proves they come from **caching, not faster raw
prefill** (vLLM actually prefills 1.55× *faster* on unique prompts). SC8 needs a
framing fix (it's a repeated/RLHF-loop win, not single-call best-of-N). SC9 is an
honest loss whose magnitude is soft.

---

## 1. Meta-rigor: is the comparison *framework* fair?

Before per-scenario, three framework-level questions.

### 1a. "sglang RadixAttention-ON vs vLLM APC-OFF" — is that a fair fight?

**Yes, because APC cannot be turned ON for vLLM on this model.** vLLM's prefix
cache (APC) forces `mamba_cache_mode='align'` for hybrid-Mamba models, which
MRV2 hard-rejects ("Model Runner V2 has not yet supported
mamba_cache_mode='align'"); MRV1 crashes; the CG path asserts
`block_size 528 > max_num_batched_tokens`. So **APC-OFF is vLLM's *only* working
configuration** here — we compare each engine's best *working* config, which is
the only fair comparison available. (See showcase blog §5 for the full root-cause.)

**Important boundary:** this means the wins are specific to *this hybrid-Mamba
model*. On a dense model where vLLM APC works, vLLM would also cache prefixes
and the gap would shrink dramatically. The wins are "sglang caches, vLLM can't
(on hybrid Mamba)" — a real, structural advantage on the cutting-edge model
class, not "sglang is universally faster."

### 1b. Are the engine configs equally optimal?

Both use their established tuned DLIN configs:

| knob | sglang | vLLM-MRV2 |
|---|---|---|
| TP / dtype / FP8 | TP4, bf16, FP8 (Q2 GEMM) | TP4, bf16, FP8 (model) |
| mem fraction | 0.55 | 0.55 |
| cuda graph (decode) | on, max_bs_decode=4 | on, capture [1,2,4] |
| max batched seqs | max_running_requests=4 | max_num_seqs=4 |
| prefill chunking | chunked_prefill_size=512 | vLLM default chunking |
| prefix cache | RadixAttention **on** | APC **off** (structural) |
| MoE kernel | fused (`invoke_fused_moe_opt`, _dl_C) | _dl_C |

These are comparable and each is the known-good DLIN config. **Residual
confounder:** I cannot fully prove both are *equally* optimal — e.g., sglang's
`chunked_prefill_size=512` may be suboptimal for M=1 prefill (SC6 shows sglang
*slower* at raw prefill, which is consistent with a chunking overhead). But note
this confound, if real, makes sglang look *worse* on raw prefill — it does **not**
inflate the caching wins.

### 1c. Is the metric measuring what we claim?

- The **ratio** (sglang-time / vLLM-time, same harness) is the valid signal —
  both engines pay the same offline-`generate` per-call overhead, so it cancels
  in the ratio.
- The **absolute tok/s** (e.g., sglang 13 tok/s in SC7) are *not* server
  throughput — they're offline-overhead-bound for short (24-token) decodes.
  sglang's online decode is ~35 tok/s. So read the **ratios**, not the absolutes.

---

## 2. The linchpin experiment: SC6 raw-prefill parity

The single most important rigor question: **are the wins from RadixAttention
caching, or also from sglang having faster raw-prefill kernels?** If sglang just
prefills faster (same `_dl_C` kernels, but better orchestration), the "caching"
story would be wrong.

**SC6** answers it: N **unique** ~1.3K-token prompts (distinct prefixes ⇒
RadixAttention *cannot* hit), + 4-token decode. Both engines prefill every
prompt fully. Distinct prompts each pass defeat cross-pass caching.

| SC6 raw prefill (no cache) | rate |
|---|---|
| **sglang** | **49 tok/s** (1337 tok × 8 in 219.8 s) |
| **vLLM-MRV2** | **76 tok/s** (1337 tok × 8 in 140.0 s) |
| → **vLLM is 1.55× *faster* at raw prefill** | |

**Conclusion:** sglang's raw prefill is **not** faster than vLLM's — it's slower.
Therefore the SC5/SC7/SC8/SC10 wins **cannot** be explained by raw-prefill speed;
they must come from **RadixAttention caching** (sglang skips the prefill for
shared prefixes; vLLM, APC-off, pays it every time). This is the clean
attribution. (Corollary: on a workload of *all-unique* prompts, vLLM wins — SC6
*is* that workload, and vLLM is 1.55× faster.)

> Side note: sglang's 49 tok/s raw prefill is suspiciously slow (both engines
> share `_dl_C`). Likely `chunked_prefill_size=512` overhead for M=1. Irrelevant
> to the caching wins — when sglang *does* cache, it only prefills the tiny
> incremental suffix (~10–50 tokens), so the slow raw-prefill rate never bites.

---

## 3. Per-scenario rigor

### SC5 — multi-user fork (sglang 8.22×) — ✅ RIGOROUS
- **Mechanism:** 2 users × 4 turns over a shared root, interleaved. sglang
  caches the root + each user's growing branch; each turn extends only the
  incremental suffix. vLLM (APC-off) re-prefills the whole growing conversation
  every turn.
- **Fairness:** same model/GPUs/prompt/decode(24)/temp(0). ✓
- **Confound check (SC6):** the 8.22× is *not* raw-prefill speed (sglang slower
  there) — it's the per-turn re-prefill vLLM pays. Confirmed caching. ✓
- **Cache-truth check:** sglang's reps decreased (15519→13215 ms) across passes,
  consistent with the radix tree warming. ✓
- **Verdict: rigorous.** Realistic workload (multi-tenant), correctly attributed.

### SC7 — long-RAG throughput (sglang 16.25×) — ✅ RIGOROUS
- **Mechanism:** ~2K-token doc × 8 sequential queries. sglang prefills the doc
  once; vLLM re-prefills ~2K tokens × 8 queries × passes.
- **Confound check (SC6):** vLLM's slow prefill (76 tok/s ⇒ ~28 s per 2K
  re-prefill) is its *raw* rate, not a config cripple (SC6 measured it on unique
  prompts). sglang avoids it via caching. ✓
- **Verdict: rigorous.** The large ratio reflects the large prefix × many
  queries — exactly where caching helps most.

### SC10 — shared system-prompt (sglang 7.05×) — ✅ RIGOROUS
- Same mechanism as SC7 (caching) with a shorter prefix (0.9K) × more tenants
  (12). SC6 confirms attribution. ✓ **Rigorous.**

### SC8 — best-of-N parallel sampling (sglang 5.84×) — ⚠️ MIXED (two factors)
- **The confound:** the SC8 harness runs warmup + 3 measured reps of the **same**
  ~0.9K prompt. sglang's RadixAttention caches that prompt after the warmup ⇒
  measured reps skip prefill (cache hit). vLLM (APC-off) re-prefills it every rep.
- **The probe (SC8b):** identical to SC8 but a **unique** prompt per call ⇒ no
  cross-call cache. Measured:

  | best-of-N (n=4, temp=0.7) | sglang | vLLM |
  |---|---|---|
  | SC8 warm (same prompt reused → sglang caches) | **22.2 tok/s** | 3.8 tok/s |
  | SC8b cold (unique prompt/call → no cache) | **5.7 tok/s** | 2.8 tok/s |

- **Decomposition of SC8's 5.84× (22.2 / 3.8):**
  - **Cross-call caching ≈ 3.9×** — sglang warm (22.2) vs sglang cold (5.7).
    This is the RLHF-rejection-sampling-loop advantage (the prompt is regenerated
    many times; sglang caches it, vLLM can't).
  - **Single-call best-of-N edge ≈ 2.0×** — sglang cold (5.7) vs vLLM cold (2.8).
    sglang handles n=4 parallel samples better than vLLM on DLIN.
- **The vLLM n=4 pathology:** vLLM's n=4 best-of-N is **2.8 tok/s** aggregate
  — vs its n=1 single-stream decode of **39.6 tok/s** (SC9). Per-sequence that's
  ~0.7 tok/s, ~50× slower than n=1. vLLM is clearly **not** batching/capturing
  n=4 decode efficiently on DLIN (CG capture is `[1,2,4]`, so batch=4 *should* be
  captured — likely a best-of-N code-path issue). So part of SC8's win is a vLLM
  n=4 weakness that may be fixable. sglang's n=4 (5.7) is also well below the
  ~4× single (≈120) one would expect from batched decode — so neither engine
  batches n=4 well, but sglang less badly.
- **Verdict:** **both factors are real sglang wins**, but they measure different
  workloads. SC8 (5.84×) is the **repeated-best-of-N / RLHF-loop** number
  (caching-dominated). SC8b (~2×) is the **single-call best-of-N** number.
  Fix: relabel SC8 as "repeated best-of-N (RLHF loop)" and surface SC8b as the
  single-call figure; flag the vLLM n=4 pathology so the single-call 2× isn't
  read as a fundamental decode advantage.

### SC9 — pure long decode (vLLM 1.30×) — ⚠️ DIRECTION RIGOROUS, MAGNITUDE SOFT
- **Mechanism:** ~14-token prompt + 128-token single-stream greedy decode.
  Negligible prefill ⇒ decode-bound. vLLM's DLIN decode-IPC edge wins.
- **Direction is robust:** vLLM 39.6 > sglang 30.5 tok/s — vLLM genuinely decodes
  faster here. ✓
- **Magnitude caveat:** sglang's 30.5 tok/s is **below its ~35 tok/s optimum**
  (measured in the tuned decode config, see `sglang-dlin-decode-gap-debug-blog`).
  The showcase config (mem 0.55, max_running_requests=4, short warmup) undershoots.
  At sglang's 35 optimum, the gap is 39.6/35 = **1.13×**, not 1.30×. So "vLLM
  1.30×" is an **upper bound**; the honest decode gap is ~1.1–1.3×.
- **Verdict:** rigorous that vLLM wins decode; report the magnitude as a range.

---

## 4. The supporting path (evidence chain)

For each conclusion, the evidence you can re-derive:

1. **"Wins are caching, not raw prefill"** ← SC6 (unique-prompt raw prefill:
   sglang 49 < vLLM 76 tok/s). Since sglang is *slower* raw, any sglang win must
   be caching. Reproduce: `compare --scenarios SC6`.
2. **"SC5/SC7/SC10 are clean caching wins"** ← SC6 (#1) + the scenario's own
   decreasing reps (cache warming) + the mechanism (shared prefix, vLLM
   re-prefills). Reproduce: `compare --scenarios SC5,SC7,SC10`.
3. **"SC8 = caching (3.9×) + single-call best-of-N edge (2×)"** ← SC8 (same
   prompt reused): sglang 22.2 vs vLLM 3.8. SC8b (unique prompt/call): sglang 5.7
   vs vLLM 2.8. sglang-warm/sglang-cold = 22.2/5.7 = 3.9× (the caching factor);
   sglang-cold/vLLM-cold = 5.7/2.8 = 2.0× (the single-call best-of-N factor).
   vLLM's n=4 (2.8 tok/s) vs its n=1 (39.6, SC9) flags an n=4 pathology.
   Reproduce: `compare --scenarios SC8,SC8B`.
4. **"SC9: vLLM wins decode, magnitude ~1.1–1.3×"** ← SC9 (vLLM 39.6 > sglang
   30.5) + the known sglang decode optimum (~35, from the decode-gap work).
5. **"APC-off is vLLM's only option"** ← trying `enable_prefix_caching=True` on
   this model → MRV2 hard-rejects `mamba_cache_mode='align'` (showcase blog §5).

---

## 5. Honest summary & fixes

**Rock-solid (report as-is):** SC5 (8.22×), SC7 (16.25×), SC10 (7.05×) — caching
wins, attribution proven by SC6.

**Needs a fix:**
- **SC8:** relabel as "repeated best-of-N (RLHF rejection-sampling loop)" — the
  5.84× is caching-dominated. Surface **SC8b (~2×, single-call)** as the
  single-best-of-N number, and **flag the vLLM n=4 pathology** (2.8 tok/s vs
  39.6 n=1) so the single-call 2× isn't mistaken for a fundamental sglang decode
  advantage — it's partly a vLLM n=4 weakness that may be fixable.
- **SC9:** state the magnitude as ~1.1–1.3× (sglang decode here is below optimum).

**Not a rigor problem, but worth noting:**
- sglang's raw prefill (49 tok/s) is slower than vLLM's (76) — likely
  `chunked_prefill_size=512` overhead for M=1. Doesn't affect caching wins (cached
  suffix is tiny), but it's a real sglang prefill inefficiency worth a separate
  look (raise chunk size / fuse M=1 prefill).

**What would change the wins:** a model where vLLM APC works (dense models) —
then vLLM caches too and the 7–16× gaps collapse to the small decode/prefill
differences. The wins are a **hybrid-Mamba structural advantage**, stated honestly.
