# SGLang vs vLLM: Structured Output & Prefix Sharing on Denglin GPUs

## A Debug Field Report — **v2 (corrected): proper vLLM package → matched FP8+cuda-graph → vLLM wins JSON**

**Date:** 2026-07-21 (v2) · **Hardware:** DLIN KS38 (32× QUAD, 32 GiB/card) · **Model:** Qwen3.5-35B-A3B-**FP8** (both engines) · **GPUs:** 28–31, TP4 · **Skill:** `/dl-compare-sglang-vllm`

> ⚠️ **Correction notice.** An earlier draft (v1, same day) concluded *"sglang beats vLLM on both features."* **That was wrong** — it used the wrong vLLM package (`venv-vllm021` = vLLM 0.21.0 + a partial `vllm-new-overlay`), which (a) forced vLLM to Int4+eager and (b) threw phantom DLIN "bugs" (gumbel/block_table/penalty) that do **not** exist with the correct DLIN stack. With the correct package (`.venv`: vLLM 0.21.1.dev2 + DLIN triton, native — **no overlay**), vLLM runs **FP8 + cuda-graph**, enabling a fair **matched-config** comparison. Result: **vLLM beats sglang on JSON**; prefix is ~tied. This v2 supersedes v1. See §1 for the package mistake and §7 for the corrected verdict.

---

## Abstract (corrected TL;DR)

Fair head-to-head, **both engines FP8 + cuda-graph, same model, TP4, DLIN**:

| | JSON tok/s (unconstr → constr) | FSM overhead | JSON valid? | Prefix warm (cache-hit) |
|---|---|---|---|---|
| **sglang FP8 + CG** | 33.7 → 31.7 | 6% | ✅ | 0.97 s |
| **vLLM FP8 + CG** (`.venv`) | **37.9 → 34.6** | 8.6% | ✅ | 1.04 s |

- **❌ JSON structured output — vLLM wins** (37.9 vs 33.7 unconstrained, ~12%; 34.6 vs 31.7 JSON, ~9%). Both emit valid schema-correct JSON. The scenario is **decode-bound** (FSM overhead negligible on both, <9%), so the engine with faster DLIN decode wins — and that's vLLM, consistent with the known sglang DLIN decode gap (`[[dlin-sglang-tp4-gpu-compute-gap]]`: sglang 27.3 ms vs vLLM 24.0 ms TP4 TPOT). **Compressed-FSM is not a differentiator on DLIN** (both use xgrammar).
- **➖ Prefix sharing — ~tied.** sglang warm 0.97 s vs vLLM 1.04 s (sglang marginally faster), but **neither shows real cache speedup** on this short-output workload (vLLM 1.00×; sglang's apparent "40×" was a one-time first-prefill JIT artifact). Workload is too decode/overhead-dominated to isolate RadixAttention vs APC.
- **Net:** at matched config, sglang does **not** beat vLLM on these features on DLIN. The durable, real findings are (1) the **sglang DLIN decode gap** (the actual lever), (2) a **sglang eager-mode correctness bug** (garbage JSON; CG fine), and (3) the **package/infra lessons** below.

---

## 1. The package mistake (why v1 was wrong) — the real debugging insight

v1 used `../venv-vllm021/bin/python` (vLLM **0.21.0**, triton **3.1.0**) plus `PYTHONPATH=vllm-new-overlay` (DLIN triton 3.3.0 + sdk-built vllm 0.21.1.dev2). Two compounding problems:

1. **Overlay doesn't reliably shadow vLLM submodules.** It overrides the top-level `vllm` package, but `vllm.v1.worker.gpu.sample.gumbel` / `...block_table` etc. resolved to the **installed 0.21.0** core, not the overlay's sdk build. So we ran 0.21.0's core against a half-applied triton overlay → `NameError: Cannot access global variable _load_ptr`, `chained boolean operators not supported`, gumbel `float64`, "double dtype". I "fixed" all four by patching `venv-vllm021`. **None of those patches are needed with the correct package** — they were artifacts of the overlay misconfiguration.
2. **FP8 + CG both failed under the overlay** → v1 was forced to compare sglang-FP8-CG vs vLLM-Int4-eager. That's not a config match; sglang's "win" was mostly *having cuda-graph while vLLM didn't*.

**The fix:** use **`.venv`**, which has the complete DLIN stack **natively installed** (no overlay):
- `vllm 0.21.1.dev2+g9511db443.sdk202607101443.cu117` (the sdk-built wheel in `whl-for-compare/`)
- `triton 3.3.0+git4b604310` (the DLIN fork)
- `_dl_C.so` + `dl_platform_plugin` / `dl_models_plugin` / `dl_quantization_plugin`

Verify: `.venv/bin/python -c "import vllm,triton; print(vllm.__version__, triton.__version__)"` → `0.21.1.dev2+g9511db443 3.3.0`. With this, vLLM loads FP8, captures cuda-graph, and generates valid output **with zero source patches**.

> **Lesson:** the canonical DLIN vLLM is the sdk-built wheel installed in `.venv` (or `whl-for-compare/`). Do **not** mix `venv-vllm021` + overlay for comparisons — the overlay is incomplete and produces misleading failures. (The `dl-compare-sglang-vllm` skill's `../venv-vllm021` path should be updated to `.venv`.)

---

## 2. Infra lessons (the debugging journey, corrected)

### 2.1 Run vLLM as a **top-level script with `if __name__ == '__main__':`**
vLLM V1 spawns EngineCore/workers via `multiprocessing` **spawn**. A worker re-imports the main module. If your script calls `LLM()` at module top level (no guard), the worker re-runs `LLM()` → `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`. Wrap all executable logic in `if __name__ == "__main__":` (`scripts/dl/vllm_features_only.py` does this).

### 2.2 `dlcc` JIT compiler needs the SDK env in the worker
At first generation, the DLIN triton JIT-compiles a kernel for the actual decode shape via **`dlcc`** (the DLIN C compiler, at `sdk-dlop-07-13-20-30/bin/dlcc`). Workers must have it on `PATH`. A subprocess wrapper that re-execs vLLM can lose the SDK env → `FileNotFoundError: 'dlcc'`. Running vLLM as a **top-level process** after `source sdk-.../env.sh` makes workers inherit the SDK env correctly.

### 2.3 GPU contention, leaked VRAM, and the wait-and-run orchestrator
The 32-card box is shared; a colleague's TP32 job repeatedly grabbed all GPUs (OOM-killing loads), and killed DLIN processes **leak VRAM**. `scripts/dl/wait_and_run_bench.sh` / `wait_run_full.sh` poll a target QUAD, confirm free over 2 checks 30 s apart, then auto-launch.

### 2.4 Transient SIGABRT under GPU pressure
Back-to-back engine loads occasionally hit `Rank 0 scheduler died (exit -6 SIGABRT)` — a GPU/driver state issue after several crashed processes, **not** a code bug. A clean GPU window (or `dlsmi -r`) fixes it.

---

## 3. Results — matched FP8 + cuda-graph (corrected)

| Config | A: JSON tok/s (unconstr→constr) | A: FSM overhead | A: JSON valid? | A: JSON sample | B: warm cache-hit |
|---|---|---|---|---|---|
| **sglang FP8 + CG** | 33.7 → 31.7 | 6% (0.940) | ✅ | `{"name": "Dr. Ada Lovelace", ...}` | 0.97 s |
| **vLLM FP8 + CG** (`.venv`) | **37.9 → 34.6** | 8.6% (0.914) | ✅ | `{"name": "Dr. Ada Lovelace", ...}` | 1.04 s |
| vLLM FP8 + eager (`.venv`) | 11.8 → 11.4 | 3% (0.966) | ✅ | (valid) | 1.93 s |

All three configs produce valid, schema-correct JSON. (The earlier sglang-eager **garbage** JSON was the sglang-side correctness bug in §6.2, not a vLLM issue.)

---

## 4. Why vLLM wins JSON on DLIN

The JSON scenario is **decode-bound**: applying the grammar mask (the Compressed-FSM work) costs <9% on **both** engines, so it's not the bottleneck. The bottleneck is raw decode throughput, and on DLIN vLLM decodes faster than sglang (known gap, `[[dlin-sglang-tp4-gpu-compute-gap]]`: identical GPU kernels, but sglang carries ~3 ms/step extra host/pipeline overhead). Therefore:

> SGLang's marketed **Compressed-FSM** advantage does **not** materialize vs vLLM on DLIN, because both already use xgrammar (cheap FSM) and the contest is decided by decode speed — where sglang is disadvantaged on this hardware.

The blog's big structured-output wins (5–10×) are vs *expensive* decoders (Guidance, outlines), not vs vLLM+xgrammar.

---

## 5. The prefix-sharing picture (murky, workload-limited)

- **vLLM-CG:** cold 1.04 s ≈ warm 1.04 s → **1.00×** (no measurable cache speedup).
- **sglang-CG:** cold 38.96 s (one-time first-long-prefill JIT) vs warm 0.97 s → apparent "40×", but that's the JIT artifact, not cache.
- Both warm latencies are close (sglang 0.97 s, vLLM 1.04 s); the 16-token output is too short for prefill (the part cache saves) to dominate. **This workload cannot cleanly separate RadixAttention from APC.** A proper prefix test needs long shared prefix + long output + concurrency (§8 P2a).

---

## 6. Real sglang-on-DLIN issues (stand, independent of the package mix-up)

### 6.1 DLIN decode gap (the actual lever)
sglang-CG 33.7 tok/s vs vLLM-CG 37.9 tok/s. Same model, same FP8, same TP4, same cuda-graph — vLLM is ~12% faster on decode. Per `[[dlin-sglang-tp4-gpu-compute-gap]]`, GPU kernels are identical; the gap is sglang's per-step host overhead (~21 eager tensor copies/step + scheduler). Closing this is what would make sglang competitive on JSON.

### 6.2 sglang eager-mode correctness bug
sglang FP8+**eager** emits **garbage** JSON (`"name":"While the text mentions "...一个人的研究科学家`) on DLIN; FP8+**cuda-graph** is correct. Same model/prompt — only the CG/eager path differs. A real sglang-on-DLIN divergence (same family as prior [[dlin-sglang-is-neox-quality-fix]] / quant bugs): the eager decode path hits a different (buggy) kernel than the CG-captured one.

---

## 7. Honest verdict (corrected)

- **JSON structured output:** vLLM **beats** sglang on DLIN at matched FP8+CG (37.9 vs 33.7 tok/s). sglang's Compressed-FSM is not a differentiator here (both xgrammar, decode-bound, sglang decode-disadvantaged).
- **Prefix sharing:** ~tied / inconclusive on this workload (neither shows clean cache speedup; workload too short).
- **The goal "validate sglang beats vLLM on both features" is NOT met on DLIN.** v1's opposite conclusion was a wrong-package artifact.
- **What IS true:** sglang runs on DLIN; vLLM 0.21.1.dev2 (`.venv`) also runs cleanly (FP8+CG) — the earlier "vLLM needs 4 patches" was wrong. sglang has a real DLIN decode gap and an eager correctness bug to fix.

---

## 8. Next-step optimization plan

The levers are now clearly **sglang-side** (the decode gap is why sglang loses JSON). Ordered by leverage:

### P0a — Close sglang's DLIN decode gap ⭐ (highest leverage)
sglang-CG 33.7 vs vLLM-CG 37.9 — gap is host overhead, not GPU kernels (`[[dlin-sglang-tp4-gpu-compute-gap]]`). Closing it makes sglang decode ≥ vLLM → sglang wins JSON at matched config (FSM overhead is a wash).
- **Action:** profile a decode step (`scripts/dl/per_step_timing.py`), enumerate the ~21 eager copies/step + scheduler overhead, eliminate/fuse. Target sglang-CG ≥ 38 tok/s.
- **Refs:** `docs/dl/sglang-vs-vllm-tp4-20260715-report.md`.

### P0b — Enable the fused-MoE fast path inside cuda-graph (PDL blocker)
sglang is forced onto slow `GEMMEX=2` MoE in CG because fast `invoke_fused_moe_opt` crashes (`per_token_group_quant_8bit_v2.cuh:396 .enable_pdl` not CG-capturable). vLLM runs CG+same `_dl_C.so` fine → it's sglang's capture state. Big decode win. `[[dlin-sglang-moe-pdl-cg-blocker]]`.

### P0c — Root-cause the sglang eager correctness bug (§6.2)
sglang eager → garbage JSON; CG → correct. Bisect which eager-path kernel diverges (attention/norm that CG replaces). Until fixed, sglang-eager is unusable for quality work.

### P1a — torch.compile Phase II (fused norm/act+quant kernels)
Inject the PostGrad pass manager so decode uses the fused kernels already in vLLM's `_dl_C.so`. Compounds with P0a/b. `[[dlin-sglang-torch-compile-phase2-plan]]`, `docs/dl/torch-compile-stage0-1-plan.md`.

### P1b — Keep the matched-CG comparison as a regression gate
vLLM FP8+CG now works on DLIN (`.venv`). Wire `scripts/dl/vllm_features_only.py` + the sglang bench into a quick CI-able regression check so sglang decode improvements are measured against vLLM at matched config (not the old Int4+eager baseline). Fix the `dl-compare-sglang-vllm` skill's vLLM path: `.venv`, not `venv-vllm021`+overlay.

### P2a — Redesign the prefix-sharing benchmark (to actually show cache benefit)
Current B is too short to separate RadixAttention from APC. Use **long shared prefix (4–8 K tok) + long output + online serving with concurrency** (the regime where RadixAttention + cache-aware router shine). Re-run via `scripts/dl/benchrun_sglang.py` (serving) with a shared-prefix dataset.

### P2b — Breadth
Validate on DeepSeek-V3 (blog flagship) and a non-hybrid dense model to confirm whether the decode-gap finding generalizes beyond Qwen3.5-35B-A3B.

---

## Reproduction (corrected — uses `.venv`, no overlay)

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh        # puts dlcc on PATH for vLLM workers

# sglang FP8 + cuda-graph:
CUDA_VISIBLE_DEVICES=28,29,30,31 SKIP_VLLM=1 MEM_FRAC=0.6 \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py

# vLLM FP8 + cuda-graph (top-level script, main-guard):
CUDA_VISIBLE_DEVICES=28,29,30,31 \
  VLLM_MODEL=/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ VLLM_DTYPE=bfloat16 VLLM_EAGER=0 \
  .venv/bin/python scripts/dl/vllm_features_only.py

# vLLM FP8 + eager (mode-matched control):
CUDA_VISIBLE_DEVICES=28,29,30,31 \
  VLLM_MODEL=/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ VLLM_DTYPE=bfloat16 VLLM_EAGER=1 \
  .venv/bin/python scripts/dl/vllm_features_only.py
```

---

## Live Results (2026-07-21, GPU 28–31, TP4, sdk-dlop, both FP8)

| Run log | Engine | A unconstr | A JSON | FSM ratio | A valid | B warm |
|---|---|---|---|---|---|---|
| `run_full.log` | sglang FP8+CG | 33.7 tok/s | 31.7 tok/s | 0.940 | ✅ | 0.97 s |
| `/tmp/vllm_fp8_cg.log` | vLLM **MRV2** FP8+CG | **37.9 tok/s** | **34.6 tok/s** | 0.914 | ✅ | 1.04 s |
| `/tmp/vllm_fp8_eager2.log` | vLLM MRV2 FP8+eager | 11.8 tok/s | 11.4 tok/s | 0.966 | ✅ | 1.93 s |
| `/tmp/vllm_mrv1.log` | vLLM **MRV1** FP8+CG | **FAIL** | — | — | — | — |

**MRV1 vs MRV2 on DLIN:** vLLM **MRV1 does not run** (`assert num_cache_lines >= batch` during warmup — a DLIN Mamba-cache assertion; a sibling of the skill's 坑③). **Only MRV2 works** → the vLLM side of any DLIN comparison is MRV2.

---

## §9. Optimization session progress (2026-07-21)

Worked the plan in §8. Status:

### Done this session
- **Fixed the sglang↔vLLM `_dl_C` double-registration SIGABRT** (the blocker that made sglang init crash intermittently). Root cause: sglang's `layernorm.py` and `fp8_utils.py` hardcoded `../venv-vllm021/.../vllm/_dl_C.so`, while `fp8.py` imports `.venv`'s vLLM → **two different paths** to the (byte-identical) `_dl_C.so` → `torch.ops.load_library` registers the `_dl_C` TORCH_LIBRARY **twice** → `c10::Error` SIGABRT at cuda-graph capture. **Fix:** both loaders now derive the path from the *importable* `vllm` (`vllm.__file__`), so a single path is used. sglang now initializes deterministically. (Files: `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`.) Note: the two `_dl_C.so` are byte-identical (md5 `adc7f6a2…`), so this is a correctness fix, not a speed change.
- **Fixed the `/dl-compare-sglang-vllm` skill** (P1b): vLLM path → `.venv/bin/python`, no overlay; FP8 for both engines; documented the `if __name__=='__main__'` + `dlcc` gotchas. (`.claude/skills/dl-compare-sglang-vllm/SKILL.md`.)
- **Built `scripts/dl/vllm_features_only.py`** — standalone in-process vLLM benchmark (top-level script + main guard) that runs FP8+CG cleanly from `.venv`.

### Investigated — P0a (decode gap) finding CORRECTS the prior hypothesis
A read-only agent traced all per-decode-step eager copies. **The prior "≈21 eager copies = the decode gap" hypothesis is wrong** — copy *count* does not explain the ~7 ms gap (sglang 33.5 ms vs vLLM 26.4 ms per token). The gap is driven by **GPU sync points** (`.item()` D2H syncs, esp. in `SGLANG_DL_MULTI_STEP` setup) and **structural IPC** (sglang's multi-process Engine scheduler round-trip vs vLLM's). The necessary cuda-graph `fill_from` copies (~6–7/step) and a few scalar allocs (`num_token_non_padded`, positions, mamba_track_mask) are real but small. ⇒ **Copy elimination is low-leverage; closing the gap needs sync-point reduction + IPC work, not tensor-copy patches.**

### Tried + found broken — `SGLANG_DL_MULTI_STEP` has a real KV memory leak
The tp4-gap report listed `SGLANG_DL_MULTI_STEP=4` as a "~2 ms" lever. Retested (post `_dl_C` fix): it crashes with `AssertionError: Unexpected overallocated KV cache, kv_committed_len=112, kv_allocated_len=208` (`mem_cache/common.py:664`). Exempting that assertion (to let multi-step run) instead surfaces `ValueError: pool memory leak detected!` — i.e. **the assertion was correctly guarding a genuine leak**: DL multi-step over-allocates KV each step but the per-request free path doesn't reclaim it. **Reverted** the exemption (don't weaken a correct safety check). ⇒ `SGLANG_DL_MULTI_STEP` is **not viable** on this build until its KV bookkeeping is fixed (the extra-step tokens must be committed/freed correctly in `tp_worker._dl_multi_step_decode` / `schedule_batch`). This is a concrete newly-filed bug, but fixing it is deeper than a one-liner.

### Deferred (multi-session, with clear next steps)
- **P0a (real):** reduce GPU sync points in the decode hot path (DL multi-step `.item()` syncs at `tp_worker.py:626-628`; carry CPU-side counters / pre-fetch on overlap stream) + attack the scheduler↔worker IPC round-trip. This is the actual lever.
- **P0b (MoE PDL CG blocker):** `[[dlin-sglang-moe-pdl-cg-blocker]]` — sglang forced to slow `GEMMEX=2` in CG; fix capture state.
- **P0c (sglang eager garbage-JSON bug):** still open — sglang FP8+eager emits garbage, CG correct. Needs a logits bisection (eager vs CG kernel path).
- **P1a (torch.compile Phase II):** own plan doc.

### Net effect measured
With the `_dl_C` fix sglang runs deterministically; the fair matched comparison stands at **sglang FP8+CG 33.7 vs vLLM MRV2 FP8+CG 37.9** (vLLM +12% on decode-bound JSON; MRV1 N/A on DLIN). No sglang speed optimization landed yet — the decode gap is structural (§9 P0a finding), so the next real lever is sync-point/IPC reduction, not copy elimination.

---

## Sources

- Official: [SGLang RadixAttention blog](https://www.lmsys.org/blog/2024-01-17-sglang/), [SGLang v0.4 (xgrammar)](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [Compressed FSM](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- Independent: [InferenceX by SemiAnalysis](https://inferencex.semianalysis.com/), [vLLM tops Artificial Analysis (2026-05)](https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis)
- In-repo: [`sglang-learning-structured-output-and-radix.md`](sglang-learning-structured-output-and-radix.md), [`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md) (decode gap)
- Memory: `dlin-sglang-tp4-gpu-compute-gap`, `dlin-sglang-moe-pdl-cg-blocker`, `dlin-sglang-torch-compile-phase2-plan`
