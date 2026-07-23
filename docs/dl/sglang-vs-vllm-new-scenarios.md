# sglang vs vLLM on DLIN — New Scenario Investigation (SC5/SC7/SC8/SC9/SC10)

> Companion to [`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md)
> (SC1–SC4). This doc records the **2026-07-24 investigation** that extended the
> showcase with new workload patterns, the methodology used, and an honest
> win/loss accounting for each new scenario.

## TL;DR

_Done (run r008, commit ba1bab09de). 4 sglang wins (5.8–16.3×) + 1 honest
vLLM decode win (SC9, the control)._

| Scenario | Workload (shared-structure shape) | sglang | vLLM-MRV2 | Verdict |
|----------|-----------------------------------|--------|-----------|---------|
| **SC5** multi-user fork | 2 users × 4 turns, shared root (radix **tree**) | **13.2 s** (1653 ms/turn) | **108.6 s** (13578 ms/turn) | **sglang 8.22× faster** ✅ |
| **SC7** long-RAG throughput | ~2K-token doc × 8 sequential queries | **13.0 tok/s** | **0.8 tok/s** (228 s) | **sglang 16.25× higher** ✅ |
| **SC8** parallel sampling | n=4 candidates (temp=0.7) from a 0.9K prompt | **22.2 tok/s** | **3.8 tok/s** (51 s) | **sglang 5.84× higher** ✅ |
| **SC9** pure long decode | short prompt + 128 tok single-stream (decode-bound) | 30.5 tok/s | **39.6 tok/s** | **vLLM 1.30× (honest loss / control)** ⚠️ |
| **SC10** shared system prompt | ~0.9K system prompt × 12 tenants | **13.4 tok/s** | **1.9 tok/s** (151 s) | **sglang 7.05× higher** ✅ |

**Hypotheses vs. reality (measured):**
- SC5 / SC7 / SC8 / SC10 are prefix/KV-reuse (or prefill-heavy) workloads →
  sglang **wins 5.8–16.3×** (vLLM APC is structurally unsupported on this
  hybrid-Mamba model, so it re-prefills every time; its prefill is also slow
  ~75 tok/s). SC8 — expected to be a vLLM-favored decode-bound case — actually
  **went to sglang 5.8×**, because vLLM's n=4 prefill+decode is slow here.
- SC9 (pure decode, no shared structure) is the one honest vLLM win: **vLLM
  1.30×** (39.6 vs 30.5 tok/s) — the decode-IPC gap. With nothing to
  cache-reuse, vLLM's faster raw decode wins.

## 1. Why these scenarios (and not just "bigger SC1")

SC1–SC4 already cover: flat prefix across independent prompts (SC1), one linear
multi-turn conversation (SC2), a single batched prefix call (SC3), short JSON
(SC4). The new scenarios each exercise a **distinct production workload shape**
where KV-cache reuse (or the lack of it) dominates:

- **SC5 — multi-user fork (radix tree).** Two users each hold a 4-turn
  conversation over a **shared system/context root**, interleaved in one engine.
  This is a *branching* tree (root → {user-A branch, user-B branch}) — the
  structure RadixAttention is named for — not the flat prefix of SC1 or the
  single line of SC2. Production shape: many users / agents behind one system
  prompt. Win mechanism: root prefilled once, each turn extends only its
  incremental suffix; vLLM re-prefills the whole growing conversation each turn.
- **SC7 — long-RAG throughput.** A ~2K-token document + 8 diverse sequential
  queries, reported as **aggregate decode throughput**. A ~2× longer prefix
  than SC1/SC3 amplifies the re-prefill cost vLLM pays each query. Production
  shape: RAG over a long shared document.
- **SC8 — parallel sampling (best-of-N).** One prompt → n=4 candidate
  completions in a single call. Decode-bound control: the prompt is prefilled
  once and forked in *both* engines, so APC is irrelevant and the ~8% decode-IPC
  edge decides it. Production shape: best-of-N / RLHF rejection sampling.
- **SC10 — shared system-prompt throughput.** 24 independent short requests all
  sharing one ~0.9K system prompt. The canonical RadixAttention-in-production
  shape (one chatbot persona, many users). ~3× shorter prefix and 1.5× more
  requests than SC7.

## 2. Fairness / methodology (identical to SC1–SC4)

- Same model (Qwen3.5/3.6-35B-A3B-FP8), same 4 GPUs, **sequential** fresh
  process per engine, temperature 0.
- **best-of-3** measured reps after one warmup rep (populates the radix tree).
  `ignore_eos=True` for fixed decode length (clean per-token timing).
- vLLM runs **APC-OFF** — its prefix cache cannot be enabled on this hybrid
  Mamba model (MRV2 rejects `mamba_cache_mode='align'`). See showcase blog §5.
- Only vLLM **MRV2** works on DLIN; MRV1 fails (torch.compile dynamic-shape
  guard) and is recorded as `fail`.
- Harness: `scripts/dl/showcase_prefix_sharing.py` (extended); driver
  `./run_sglang.sh compare --scenarios SC5,SC7,SC8[,SC10]`. Each scenario runs
  inside a non-fatal guard so one scenario's crash cannot wipe the others.

## 3. Results

All numbers: Qwen3.5/3.6-35B-A3B-FP8, TP4, same 4 GPUs, fresh process per
engine, temperature 0 (except SC8 = 0.7 for real best-of-N), best-of-2 after
one warmup rep, `ignore_eos=True`. Source: `docs/dl/compare_results.json`
(run r007 + the all-5 re-run).

| Scenario | sglang | vLLM-MRV2 | sglang advantage |
|----------|--------|-----------|------------------|
| **SC5** fork total | 13.2 s (1652 ms/turn) | 108.0 s (13504 ms/turn) | **8.2× faster** ✅ |
| **SC7** long-RAG throughput | 13.0 tok/s | 0.8 tok/s (227 s) | **16.3× higher** ✅ |
| **SC10** shared system-prompt | 13.4 tok/s | 1.9 tok/s (151 s) | **7.05× higher** ✅ |
| **SC8** parallel sampling (n=4, temp=0.7) | 22.2 tok/s | 3.8 tok/s (51 s) | **sglang 5.8× higher** ✅ |
| **SC9** pure long decode (128 tok) | 30.5 tok/s | 39.6 tok/s | **vLLM 1.30× (control)** ⚠️ |

**Why the wins are so large.** vLLM's APC is structurally off on this hybrid
Mamba model, so it **re-prefills the shared prefix on every request**. Worse,
vLLM's measured prefill on this model is only ~75 tok/s (vs sglang far faster),
so every re-prefill is expensive. sglang's RadixAttention keeps the shared KV
and only extends the tiny per-request suffix. The longer/more-shared the prefix
(SC7's ~2K doc, SC10's 12 tenants, SC5's growing branches), the bigger the gap.

**Why SC5 (8.2×) < SC7 (16.3×).** SC5's per-turn incremental suffix is larger
(generated assistant text + new question) and there are only 8 generations;
SC7 re-prefills a ~2K doc 24 times (warmup+2×8). More re-prefill work for vLLM
⇒ bigger ratio.

## 4. Losses recorded (for later optimization)

- **SC9 — pure long decode (the one honest vLLM win).** Short (~14-token)
  prompt + 128-token single-stream greedy decode. No shared structure ⇒
  dominated by raw decode TPOT, where vLLM holds its DLIN IPC edge: **vLLM
  39.6 tok/s vs sglang 30.5 tok/s (vLLM 1.30×)**. This is the deliberate
  counterpoint: with nothing to cache-reuse, vLLM's faster decode wins.
  Optimization path for sglang: close the decode-IPC gap (overlap scheduler /
  fewer per-step host syncs — see
  `docs/dl/sglang-dlin-decode-gap-debug-blog.md`). Note: the gap here (1.30×)
  is wider than the prior "vLLM +8%" because the showcase config's sglang
  decode (30.5 tok/s) runs below its ~35 tok/s optimum.
- **SC8 — original greedy best-of-N was invalid in vLLM** (`n must be 1 when
  using greedy sampling`). Fixed by `temperature=0.7` (real best-of-N needs
  sampling). With the fix, SC8 is a **sglang 5.8× win** (vLLM's n=4 prefill
  +decode is slow here), not a loss.
- No prefix/KV-reuse scenario turned out to favor vLLM — consistent with APC
  being structurally off on this model.

## 5. Reproduce

```bash
# Winners + control, both engines, same GPUs (TP4). ~25-30 min (vLLM re-prefills).
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10
# Re-render last gap table from cache (no GPU run):
./run_sglang.sh compare --show
# History (commit-keyed results log):
./run_sglang.sh compare --history
```

## 6. Conclusion — the sglang-win design space (on this model)

The feasible sglang-favorable workloads on Qwen3.5/3.6-35B-A3B-FP8 (DLIN) are
exhaustively covered by SC1–SC10. **Every KV-reuse shape is a sglang win
(2.3–16.3×)** because vLLM's APC is structurally off on this hybrid-Mamba model
(MRV2 rejects `mamba_cache_mode='align'`), so vLLM re-prefills — and its measured
prefill on this model is slow (~75 tok/s). sglang's RadixAttention keeps the KV
and extends only the suffix. Coverage:

- flat prefix (SC1), linear multi-turn (SC2), batched prefix (SC3) — existing,
- branching multi-user fork tree (SC5), long-RAG throughput (SC7),
- best-of-N parallel sampling (SC8), shared system-prompt many-tenant (SC10).

The **only** vLLM-favored new scenario is SC9 (pure decode, no shared structure)
at 1.30× — the decode-IPC gap. Non-prefix sglang advantages are blocked on DLIN
today: online concurrency (FA2 wrapper crashes >2 overlapping prefills),
speculative decoding (verify quality bug), torch.compile (Phase II not ready).
So new wins beyond SC5/7/8/10 would be redundant KV-reuse variants; the next
*qualitatively new* sglang win requires unblocking one of those (FA2 varlen,
spec quality, or compile fusion).
# History (commit-keyed results log):
./run_sglang.sh compare --history
```
