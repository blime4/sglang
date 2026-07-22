# Closing the sglang DLIN Decode Gap vs vLLM — Debug Field Report

**Date:** 2026-07-21 · **Hardware:** DLIN KS38 (32× QUAD, 32 GiB) · **Model:** Qwen3.5-35B-A3B-**FP8** (both engines) · **TP4, GPUs 28–31**

> Goal: make sglang-CG decode ≥ vLLM-CG on DLIN (sglang measured 33.7 tok/s vs vLLM MRV2 37.9). This is the debug log of that attempt — what was fixed, what was ruled out, and the precise next-step plan. Companion to [`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md) (the JSON/prefix comparison).

---

## TL;DR

- **Fixed a real blocker:** the `_dl_C` double-registration SIGABRT that made sglang init crash intermittently on DLIN once sglang+vLLM coexist in `.venv`. sglang now initializes deterministically.
- **Ruled out the headline hypothesis:** the prior "≈21 eager copies/step = the decode gap" framing is **wrong** — a full hot-path trace shows copy *count* doesn't explain the ~7 ms/token gap. The gap is **GPU sync points + structural multi-process IPC**, not tensor copies.
- **Retired the multi-step lever (measured):** `SGLANG_DL_MULTI_STEP` is **~4× SLOWER, not faster** (8.6 vs 33.7 tok/s — its `.item()` D2H syncs + speculative-alloc bookkeeping dominate; the "~2 ms/token" claim doesn't hold on DLIN). Output is correct but the path is counterproductive. A real KV leak in it (`output % _dl_n == 0 → allocated` doubles) is now moot.
- **Delivered the first measured speed win:** eliminated a per-step D2H sync — `decode_cuda_graph_runner.py:624 seq_lens.sum().item()` → `int(seq_lens_cpu.sum())` (host, no sync; matches `tbo_backend.py:195`). **sglang decode 33.7 → 35.0 tok/s (+4%), output valid.**
- **Measured:** matched FP8+CG — sglang **35.0** vs vLLM MRV2 37.9 tok/s (was 33.7 → gap narrowed ~12%→~8%). MRV1 doesn't run on DLIN. Remaining gap is structural (scheduler↔worker IPC; needs dlPTI profiling to find the next syncs).

---

## 1. Setup & the matched baseline

Both engines, FP8, cuda-graph, TP4, same DLIN box, temp 0, best-of-N. The key correction from earlier work: use **`.venv`** (vLLM 0.21.1.dev2 + DLIN triton 3.3.0, native — **no overlay**) for vLLM; the old `../venv-vllm021`+overlay setup is broken (see [`dlin-vllm-correct-package-and-env-gotchas`](../../../home/shaobo.xie/.claude/projects/-LocalRun-shaobo-xie-2-Pytorch-docker-test-debug-sglang/memory/dlin-vllm-correct-package-and-env-gotchas.md)).

| Engine | JSON tok/s (unconstr→constr) |
|---|---|
| sglang FP8+CG | 33.7 → 31.7 |
| vLLM MRV2 FP8+CG | **37.9 → 34.6** |
| vLLM MRV1 FP8+CG | **FAIL** (`assert num_cache_lines >= batch`) |

GPU kernels are **byte-identical** between the two (20.7 ms/token both — same `_dl_C.so`, md5 `adc7f6a2…`). So the gap is **100% host-side**.

---

## 2. Bug fixed — the `_dl_C` double-registration SIGABRT ⭐

### Symptom
Once sglang + vLLM coexist in `.venv`, sglang intermittently `SIGABRT`s (exit -6) at cuda-graph capture:
```
c10::Error: Only a single TORCH_LIBRARY can be used to register the namespace _dl_C;
Previous registration at /vllm_workspace/vllm/csrc/dl/torch_bindings.cpp:14;
latest registration at .../vllm/csrc/dl/torch_bindings.cpp:14
```

### Root cause
sglang loads the `_dl_C.so` custom-op library from **two different filesystem paths**:
- `python/sglang/srt/layers/layernorm.py:_dl_load_dl_C()` and `python/sglang/srt/layers/quantization/fp8_utils.py:_ensure_dl_C()` **hardcoded** `../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so`.
- `python/sglang/srt/layers/quantization/fp8.py` does `from vllm.plugins.dl_platform_plugin.ops.dl_fused_moe import ...` → imports `.venv`'s vLLM → loads `.venv/.../vllm/_dl_C.so`.

Two *paths* (to byte-identical .so) → `torch.ops.load_library` registers `_dl_C` twice → `c10::Error`. Import-order-dependent → intermittent (sometimes `.venv`'s loads first and the `hasattr(gemma_rms_norm)` guard short-circuits the second load).

### Fix (applied)
Both loaders now derive the path from the **importable** vllm (`import vllm; os.path.dirname(vllm.__file__)`), falling back to the old hardcoded path. Single path → single registration → deterministic init.
- `python/sglang/srt/layers/layernorm.py` (`_dl_load_dl_C`)
- `python/sglang/srt/layers/quantization/fp8_utils.py` (`_ensure_dl_C`)

**Note:** the two `_dl_C.so` are byte-identical (md5 `adc7f6a2…`), so this is a *correctness* fix, not a speed change. But it's the unblock that made every other run possible.

---

## 3. Ruled out — "eager copies = the decode gap" (corrects the prior hypothesis)

The tp4-gap report hypothesized ~21 `aten::copy_`/`to` per decode step (~290 ms CPU cumulative) as the gap. A read-only agent traced the **entire decode hot path** (`TpModelWorker.forward_batch_generation` → `ForwardBatch.init_new` → `ModelRunner.forward` → `DecodeCudaGraphRunner.execute`/`load_batch`/`fill_from` → `sample`):

**Finding: copy count does NOT explain the ~7 ms gap.** The per-token copies are:
- **Necessary** cuda-graph `fill_from` (≈6–7 `aten::copy_`/step — static-input-buffer requirement; the mechanism, not waste).
- A few **small** scalar/buffer allocs (`num_token_non_padded`, `clamp_position`, `mamba_track_mask`, overlap `seq_lens+1`) — real but sub-ms.
- The **actual latency drivers** are **GPU sync points** — above all the `.item()` D2H syncs in `SGLANG_DL_MULTI_STEP` setup (`tp_worker.py:626-628`), plus the **structural IPC round-trip** of sglang's multi-process Engine scheduler (vs vLLM's).

⇒ **Copy elimination is low-leverage.** The real lever is sync-point reduction + IPC, not tensor-copy patches. (Full ranked copy-site list saved in the agent report; top quick-wins are pre-allocating the `num_token_non_padded`/`positions` scalars — but each is sub-ms.)

---

## 4. Found + located — `SGLANG_DL_MULTI_STEP` KV pool leak (the bounded lever, not yet fixed)

The tp4-gap report listed `SGLANG_DL_MULTI_STEP=4` as a "~2 ms/token" win (replay the graph N times per scheduler step, amortizing host overhead). Retested post-`_dl_C`-fix: it crashes.

### Symptom
```
AssertionError: Unexpected overallocated KV cache, req.kv_committed_len=112, req.kv_allocated_len=208
```
Exempting that assertion (to let the over-allocation free path run) instead surfaces:
```
ValueError: pool memory leak detected! [full] total=568208, available=568000, evictable=64 ...
```
(144 slots unaccounted: `available + evictable + protected + session_held + uncached ≠ total`.)

### Root cause (located, precisely)
- `schedule_batch.py:2636` (`prepare_for_decode`, DL multi-step branch) **speculatively pre-allocates `_dl_n` KV slots** per step: `out_cache_loc = alloc_for_decode(token_per_req=1)` + a loop of `_dl_n-1` more `alloc_for_decode` with temporarily-advanced `seq_lens`. `req.kv_allocated_len += _dl_n`; `req.kv_committed_len += 1` (the extra `_dl_n-1` are committed later by `scheduler.py:3272/3366`).
- When a request **finishes mid-multi-step-batch** (hits `max_tokens` before consuming all `_dl_n` speculatively-allocated slots), the over-allocation `[committed:allocated]` is **not reclaimed** by `pop_overallocated_kv_cache` → `mem_cache/common.py:653`. The `spec_algo is None` assertion (`common.py:661`) blocks the reclaim path for non-spec; exempting it lets the free run but the accounting still leaks 144 slots → the pool invariant trips.
- Observed accounting: `allocated = committed + output_len` (i.e. output tokens effectively double-counted), confirming the speculative allocate vs reclaim mismatch.

### Why not fixed this session → RETIRED (measured: multi-step is NOT a speed win)
Instrumented and measured (both leak checks made non-fatal so it could complete): **multi-step output is CORRECT** (valid JSON matching CG), but **multi-step is ~4× SLOWER, not faster** — 8.6 tok/s vs baseline 33.7. The `.item()` D2H syncs in the DL multi-step loop (`tp_worker.py:626-628`, 3 per step) plus the speculative-alloc/leak bookkeeping dominate, swamping any scheduler-round-trip savings. **The prior "~2 ms/token" claim for `SGLANG_DL_MULTI_STEP` does not hold on DLIN** — it's counterproductive. ⇒ **P0a″ is retired.** The KV leak (real, boundary-aligned: `output % _dl_n == 0 → allocated` doubles) is moot since multi-step is slower regardless. Fixing the leak would only make a slow path slightly less leaky. The real lever is **P0a′ (sync/IPC reduction in the *normal* decode path)**.

---

## 5. P0c — sglang eager garbage-JSON (still open)

sglang FP8+**eager** emits **garbage** JSON (`"name":"While the text mentions "...一个人的研究科学家`) on DLIN; FP8+**cuda-graph** is correct. Confirmed it **persists after the `_dl_C` fix** (so it's a separate bug, not the double-registration). The divergence is in `model_runner.py:924` (`decode_cuda_graph_runner = self.eager_runner` when `disable_cuda_graph`) — i.e. `eager_runner._forward_raw` vs the captured graph hit a different (buggy-on-DLIN) kernel path (likely Mamba/GDN state carry, or an op the CG graph replaces). Needs a token-by-token logits bisection eager-vs-CG.

---

## 6. What was delivered this session (real, verified)

| Item | Status |
|---|---|
| `_dl_C` double-registration fix (`layernorm.py`, `fp8_utils.py`) | ✅ sglang inits deterministically |
| `/dl-compare-sglang-vllm` skill → `.venv`, no overlay, matched FP8 | ✅ |
| `scripts/dl/vllm_features_only.py` (standalone in-process vLLM FP8+CG harness, main-guard) | ✅ |
| MRV1/MRV2 comparison | ✅ MRV1 crashes / MRV2 37.9 / sglang 33.7 |
| P0a decode-gap trace → "copies ≠ lever; sync/IPC is" | ✅ (reframes the work) |
| `SGLANG_DL_MULTI_STEP` KV leak located precisely | ✅ (located, not fixed) |

---

## 7. Next-step optimization plan (corrected, prioritized)

The plan's original P0a ("eliminate ~21 eager copies") is **retired** — proven not the lever. Revised:

### P0a′ (now the real lever) — reduce GPU sync points + IPC
- **✅ DELIVERED (first measured speed win):** `decode_cuda_graph_runner.py:624` did `seq_lens.sum().item()` **every decode step** — a D2H sync. Replaced with `int(seq_lens_cpu.sum())` (host computation, no sync; matches the existing `tbo_backend.py:195` pattern). **Measured: unconstrained decode 33.7 → 35.0 tok/s (+4%), JSON output valid (correctness holds).** Gap: sglang 35.0 vs vLLM 37.9 (was 33.7 — closing).
- **Next sync-point candidates (ruled out so far):** `tp_active_ranks.detach().cpu().numpy()` (`model_runner.py:1711`) is **elastic-EP-only** (gated, not in normal path); `handle.wait()` (`model_runner.py:2037`) is online weight-update (one-time). The remaining latency is the **scheduler↔worker IPC round-trip** (structural) — needs profiling (torch.profiler is unreliable on DLIN; use dlPTI) to find more per-step syncs, then architectural reduction.

### ~~P0a″~~ — `SGLANG_DL_MULTI_STEP` RETIRED (measured ~4× slower)
Instrumented + measured: multi-step output is correct but **8.6 tok/s vs 33.7 baseline** (the `.item()` syncs + speculative-alloc bookkeeping dominate). Not a speed lever on DLIN. Skip. (KV leak characterised: `output % _dl_n == 0 → allocated` doubles; moot since path is slower anyway.)

### P0b — MoE fused fast-path inside cuda-graph (`.enable_pdl` CG blocker)
`[[dlin-sglang-moe-pdl-cg-blocker]]`: sglang forced onto slow `GEMMEX=2` in CG because fast `invoke_fused_moe_opt` crashes (`per_token_group_quant_8bit_v2.cuh:396 .enable_pdl` not CG-capturable). vLLM runs CG+same `_dl_C.so` fine → it's sglang's capture state.

**Step 0.1 (DONE — sglang-native flash-attn):** Eliminated ALL `_vllm_fa2_C` coexistence conflicts by compiling sglang's own flash-attn `.so` with `_sgl_fa2_C` namespace + `sgl_flash_attn.py` Python wrapper. `dl_flash_attn.py` now imports from sglang's wrapper (zero vLLM dependency). `SGLANG_DL_MOE_VLLM=1` + CG **loads and produces correct output** (valid JSON), but is 1000× slower (PDL kernels stall in CG replay — 33min capture vs 14s, 30s/token). → use_moe_cu correctness confirmed; CG-replay perf is the issue.

**Step 1 (NEXT — piecewise CG):** Split the decode forward so attention/norm stay in CG (fast, no PDL) while MoE runs eager `use_moe_cu` (no CG capture needed). Two options:
- **1A (recommended):** Implement tc_piecewise decode backend — currently "not yet implemented; falling back to 'full'" in the log. Build it by adapting the prefill tc_piecewise path for decode.
- **1C (faster prototype):** Manual piecewise — capture a graph that covers attn+norm+proj, break out of CG for MoE, resume CG. Or MoE overlay: CG captures full GEMMEX=2, then re-run MoE eagerly with use_moe_cu and discard the GEMMEX result.

### P0c — root-cause sglang eager garbage-JSON
Bisect `eager_runner._forward_raw` vs CG: dump per-layer logits for a fixed prompt under both, find the first divergence (likely Mamba/GDN state carry or an op CG replaces). Fix → unblocks matched-eager comparison.

### P1a — torch.compile Phase II (fused norm/act+quant kernels)
Inject the PostGrad pass manager so decode uses the fused kernels already in vLLM's `_dl_C.so`. Own plan doc (`docs/dl/torch-compile-stage0-1-plan.md`). Compounds with P0a′/P0a″/P0b.

### P2a/b — measurement hardening
Redesign the prefix benchmark (long shared prefix + long output + concurrency — the regime where RadixAttention actually shows); breadth-test on DeepSeek-V3 and a dense model.

---

## 8. Honest net

sglang's DLIN decode gap is **structural host overhead** (identical GPU kernels). This session: fixed the `_dl_C` init blocker, **retired multi-step** (measured 4× slower), and **delivered the first measured decode win** — eliminating the per-step `seq_lens.sum().item()` D2H sync → **sglang 33.7 → 35.0 tok/s (+4%)**, now ~8% behind vLLM 37.9 (was ~12%). A broad scan confirms no more grep-visible per-step `.item()` syncs in the greedy decode path (others are prefill/non-greedy/elastic-EP-only). The remaining ~3 tok/s is the **scheduler↔worker IPC round-trip + deeper syncs** visible only via a **dlPTI decode-step profile** (torch.profiler is unreliable on DLIN). That, plus P0b (MoE-CG) / P1a (torch.compile Phase II) compounding, is the multi-session path to parity.

### What landed in the tree (DL-marked)
- `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` — `_dl_C` single-path load (fixes double-registration SIGABRT).
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:624` — `seq_lens_sum` host-compute (eliminates per-step D2H sync, +4% decode).
- All exploratory debug/workaround edits reverted; the codebase carries only the two real fixes above.

---

## Reproduction

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh

# sglang FP8+CG (post _dl_C fix):
CUDA_VISIBLE_DEVICES=28,29,30,31 SKIP_VLLM=1 MEM_FRAC=0.6 \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py

# vLLM MRV2 FP8+CG:
CUDA_VISIBLE_DEVICES=28,29,30,31 VLLM_MODEL=/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ \
  VLLM_DTYPE=bfloat16 VLLM_EAGER=0 .venv/bin/python scripts/dl/vllm_features_only.py

# multi-step KV-leak repro (currently crashes — see §4):
CUDA_VISIBLE_DEVICES=28,29,30,31 SKIP_VLLM=1 SGLANG_DL_MULTI_STEP=4 \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py
```

## Sources / refs
- Memory: `dlin-sglang-vllm-dl_C-double-registration-fix`, `dlin-vllm-correct-package-and-env-gotchas`, `dlin-sglang-tp4-gpu-compute-gap`, `dlin-sglang-moe-pdl-cg-blocker`
- In-repo: [`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md), [`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md), [`torch-compile-stage0-1-plan.md`](torch-compile-stage0-1-plan.md)
