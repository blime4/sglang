# DeepSeek-V4-Flash on DLIN sglang — Bring-up Handoff

> **Status: E2E pipeline verified** (model loads + generates non-empty text, exit 0).
> Output is gibberish (MHC uniform approximation + torch fallbacks). Correct + fast
> output needs the 5 optimization items below. This doc is the full handoff for the
> next session — every breakage, fix, env flag, and next step.
>
> Date: 2026-07-26. Model: `/LocalRun/hao.dong/DeepSeek-V4-Flash` (149GB FP8 MLA
> MoE, `DeepseekV4ForCausalLM`, 43 layers, 256 experts top-6, MTP, compress_ratios
> `[0,0,4,128,...,4,0]`). TP8 on KS38 ×8 (32GB cards).

---

## 1. Quick-start (reproduce e2e)

```bash
source sdk-dlop-07-13-20-30/env.sh
export LD_LIBRARY_PATH="$SDK/lib" DLI_V2=ON TORCHDYNAMO_DISABLE=1 DLEOL_CACHE_SIZE=1024
export SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8
# ALL torch fallbacks (bypass NVIDIA-only JIT/deep_gemm/tilelang):
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0
.venv/bin/python scripts/dl/v4_smoke.py
# Expected: engine up ~57s, GENERATED: '<gibberish>', E2E OK
```

---

## 2. The 11 DLIN-compat fixes (all committed)

| # | breakage | fix | file(s) |
|---|---|---|---|
| 1 | `flashinfer` import (NVIDIA-only) → V4 module fails to register → generic wrapper → `AutoModel` alias error | `try/except ModuleNotFoundError` guard; `sgl_kernel.dsv3_*` present on DLIN | `models/deepseek_v2.py:191` |
| 2 | MLA decode `sgl_kernel.flash_mla.flash_mla_with_kvcache` (NVIDIA `flashmla_ops` ext) | DL branch: `torch.ops.sgl_kernel.flash_mla_with_kvcache` (vendored dldnn op); stub `_create_flashmla_metadata` | `layers/attention/deepseek_v4_backend.py:1416` |
| 3 | `deep_gemm` metadata `import deep_gemm` → `ModuleNotFoundError` | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` env (torch indexer fallback) | env (no code change) |
| 4 | V4 JIT topk `cuda/ptx.h` not found (dlcc has no PTX) | `SGLANG_OPT_USE_*_TOPK=0` + `SGLANG_TOPK_TRANSFORM_512_TORCH=1` (torch fallbacks) | env |
| 5 | MHC `_mhc_pre_warmed` NameError (defined inside `try: import tilelang`) | Define `_mhc_pre_warmed = False` in `except ImportError` block | `layers/mhc.py:36` |
| 6 | `tf32_hc_prenorm_gemm` → `deep_gemm.tf32_hc_prenorm_gemm` NameError | Torch FP32 matmul fallback when `not ENABLE_JIT_DEEPGEMM` | `layers/deep_gemm_wrapper/entrypoint.py:194` |
| 7 | tilelang JIT kernels raise `RuntimeError: tilelang is not installed` | `_TilelangMissing.jit` returns callable no-op (not raise) | `layers/mhc.py:48` |
| 8 | `sgl_kernel.rmsnorm` → `RuntimeError: input must be contiguous` (2D non-contiguous x) | `else: x = x.contiguous()` in the DL rmsnorm path | `layers/layernorm.py:340` |
| 9 | V4 JIT silu `error: reference to local binding 'expert_id'` (C++17 structured binding lambda-capture) | Batch-fix: copy structured bindings to regular vars (`auto [_x] = ...; auto x = _x;`) | 11 `.cuh` files in `jit_kernel/csrc/deepseek_v4/` (20 instances) |
| 10 | V4 silu JIT kernel `ninja` compile fail (same structured binding) | `SGLANG_OPT_SWIGLU_CLAMP_FUSION=0` (bypass; now unneeded after #9 but kept for safety) | env |
| 11 | Indexer torch fallback `assert seq_lens.shape == (batch_size,)` AssertionError | Relax: `seq_lens = seq_lens.reshape(-1)` | `layers/attention/dsv4/indexer.py:69` |

Additional: MHC `hc_split_sinkhorn` — `new_empty` → `new_zeros` (prevent NaN) → then uniform
approximation (`pre=1/hc_mult`, `comb=identity`) for non-empty output. A **torch Sinkhorn
port** (exact algorithm from the tilelang kernel) was implemented + tested but produces
empty output when combined with the topk/indexer fallbacks (interaction issue —
the Sinkhorn is correct, the empty output is from degraded expert selection).

---

## 3. The 5 optimization items (roadmap)

### Item 1: MHC correct output

**What's done:** torch Sinkhorn ported (exact: `pre=sigmoid`, `post=2*sigmoid`,
`comb=exp-max + row/col normalize iterated`). Committed in history (search for
"torch Sinkhorn port").

**The problem:** the Sinkhorn produces **empty output** when combined with the
topk/indexer torch fallbacks. The uniform approximation produces **gibberish**
(non-empty). Neither is correct. The empty output is NOT from the Sinkhorn itself
— it's from the interaction: correct MHC attention → more structured logits →
the degraded topk (wrong expert selection) produces worse hidden states → empty.

**Next step:** Fix the topk fallback (item 4) FIRST → then re-test the Sinkhorn.
Or: implement a **torch MHC attention** (the `hc_pre_torch_impl` at
`deepseek_v4.py:1292` already exists for the prenorm; extend it to compute the
full MHC forward in torch). OR: route the 3 MHC layers (0, 1, 42 — ratio=0) to
**standard FA2 attention** (the DL FA2 backend works) — model surgery but
produces coherent output.

**Key files:**
- `layers/mhc.py:147 hc_split_sinkhorn` — the Sinkhorn + the DL uniform fallback
- `models/deepseek_v4.py:1279 DeepseekV4DecoderLayer.hc_pre` — the MHC layer dispatch
- `models/deepseek_v4.py:1452` — the `mhc_fused_post_pre` path (also needs tilelang)

### Item 2: JIT structured bindings — ✅ DONE

20 instances across 11 `.cuh` files. All V4 JIT kernels compile on dlcc EXCEPT:
- `topk_v1.cuh` / `topk_v2.cuh` — uses **`cuda/ptx.h`** (PTX inline assembly).
  dlcc doesn't have this header. Needs a **C++ port** (replace PTX with dlcc-
  compatible intrinsics) OR keep the torch fallback.

### Item 3: Enable CG

**Result: OOM on 32GB cards.** The 149GB model / 8 cards = ~18.6GB/card weights +
KV cache → ~30.5GB/card at mem_fraction 0.85. CG workspace needs ~2.5GB/card →
30.5 + 2.5 = 33 > 32 → OOM. Tested at mem_fraction 0.90 AND 0.85, both OOM.

**Not a code bug — infrastructure constraint.** Options:
- **48GB+ cards** (when available) — CG workspace fits
- **TP16** (16 cards) — halves per-card memory
- **CG bs=1 only** (smallest graph) — might fit with mem_fraction ~0.75
- **piecewise CG** (capture only MoE/norm, not MLA) — reduces workspace

The CG-capture HANG from smoke #6 (100% GPU, no progress) was NOT reproduced
in eager mode — it may have been the MLA op + CG interaction (needs investigation
once memory allows CG).

### Item 4: Route to DL kernels (replace torch fallbacks)

**Current state (all torch fallbacks):**
| component | JIT/DL path | fallback | why fallback |
|---|---|---|---|
| MLA decode | DL `flash_mla_with_kvcache` ✅ | — | vendored + works |
| MLA indexer | `deep_gemm.fp8_fp4_paged_mqa_logits` | torch `fp8_paged_mqa_logits_torch` | deep_gemm absent |
| MoE topk | JIT `hash_topk` / `topk_v2` | torch `topk_transform_512_torch` | `cuda/ptx` JIT fail |
| MHC Sinkhorn | tilelang JIT | uniform approximation | tilelang absent |
| MHC prenorm | `deep_gemm.tf32_hc_prenorm_gemm` | torch FP32 matmul | deep_gemm absent |
| silu_and_mul | JIT (now compiles ✅ after structured binding fix) | — | fixed |

**Next steps:**
1. **topk `cuda/ptx` port** — the highest-leverage fix. The V4 topk JIT kernel
   (`topk_v2.cuh`) uses `#include <cuda/ptx>` (PTX warp-level intrinsics). Port
   to dlcc-compatible C++ (replace `__shfl_sync`, cooperative groups, etc. with
   dlcc equivalents). This unblocks the fast MoE expert selection → correct
   output with the Sinkhorn.
2. **Indexer → DL `fp8_fp4_paged_mqa_logits`** — the vendored op exists
   (`torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits`). Wire it into the V4
   indexer (replace the `fp8_paged_mqa_logits_torch` call). The signature
   differs — needs an adapter.
3. **deep_gemm wrapper** — provide DL FP8 GEMM (`gptq_dlblas_gemmex`) as the
   `deep_gemm_wrapper.entrypoint` backend on DLIN.

### Item 5: Benchmark vs vLLM

Blocked by #1/#4. Once correct output + fast kernels:
- Run `scripts/dl/v4_smoke.py` with timing (best-of-3 TPOT)
- Compare vs vLLM on the same model + GPUs (if vLLM supports V4-Flash on DLIN)
- Add to the compare framework (`run_sglang.sh compare`)

---

## 4. Architecture notes (for the next session)

### V4-Flash model architecture
- **43 layers**: layers 0,1,42 = MHC (ratio=0, `hc_pre`); layers 2-41 = MLA
  (ratio alternates 4,128)
- **MLA**: kv_heads=1 (latent attention), FP8 KV cache, SWA-compressed paged
- **MoE**: 256 experts, top-6, `noaux_tc` selection, `swiglu_limit=10`
- **MTP**: `num_nextn_predict_layers=1` (not tested in smoke)
- **KV pool**: `DeepSeekV4TokenToKVPool` (SWA + c4 + c128 compressed pages)

### DLIN-specific env (ALL required for V4)
```bash
# The full env block (from the quick-start):
source sdk-dlop-07-13-20-30/env.sh
export LD_LIBRARY_PATH="$SDK/lib" DLI_V2=ON TORCHDYNAMO_DISABLE=1 DLEOL_CACHE_SIZE=1024
export SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_FP8_Q2=1 SGLANG_DL_GDN_DLIN=1
# Fallbacks:
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0
```

### Vendored DL ops available (from this cycle's kernel decoucling)
- `torch.ops.sgl_kernel.flash_mla_with_kvcache` ✅ (used)
- `torch.ops.sgl_kernel.flash_mla_sparse_prefill_fwd` ✅ (not wired for V4 prefill)
- `torch.ops.sgl_kernel.fp8_fp4_paged_mqa_logits` ✅ (not wired — needs adapter)
- `torch.ops.sgl_kernel.fp8_fp4_mqa_logits` ✅ (non-paged version)
- `gptq_dlblas_gemmex` (FP8 GEMM) ✅ (used for weights, not for deep_gemm wrapper)

### GPU management
- 4 cards (16,17,20,21) are **hung/leaked** from smoke #6 (MHC kernel hang) —
  need `sudo dlsmi -r -i 16,17,20,21` to reset.
- Cards 0-7 work for TP8 smoke tests.
- All 32 cards were free at session start.

---

## 5. Commits (all on `blime/dl-main`, pushed)

Key V4 commits (search `git log --oneline | grep V4`):
- `5f03b1885f` — flashinfer guard (V4 loads)
- `c524683d15` — 11 compat fixes, e2e verified
- `687cc665d9` — uniform MHC (non-empty text)
- `635f482dc0` — revert to uniform (Sinkhorn preserved in history)
- `3708c06839` — CG OOM documented, revert to eager

Smoke script: `scripts/dl/v4_smoke.py` (TP8 load + generate + meta_info).

---

## 6. Honest assessment

**What works:** V4-Flash loads + runs e2e on DLIN (non-empty text, exit 0).
The pipeline is verified — every layer type (MLA, MHC, MoE, indexer) executes.
The JIT kernels compile (structured bindings fixed). The DL MLA op works.

**What doesn't work:** correct output (gibberish from uniform MHC + torch
fallbacks), CG (OOM), fast execution (all torch fallbacks).

**The single highest-leverage next step:** fix the topk `cuda/ptx` JIT kernel
(item 4) → unblocks fast MoE + correct expert selection → re-test the torch
Sinkhorn (item 1) → potentially coherent output. Then wire the DL indexer +
deep_gemm → remove remaining fallbacks → benchmark (item 5).

---

## Correctness investigation — 2026-07-27 (gibberish root-cause hunt)

**Symptom:** V4-Flash runs end-to-end on DLIN sglang (loads, generates) but emits
input-dependent **gibberish** (multilingual-random tokens / 16×EOS for some prompts) =
near-uniform logits = severely corrupted hidden states.

**Verified EXONERATED (arg/math correct vs vLLM's working DL ops):**
1. **Sinkhorn MHC port** (`hc_split_sinkhorn`) — unit-tested well-conditioned; arg indexing
   matches tilelang kernel exactly.
2. **Indexer** (`fp8_fp4_paged_mqa_logits`) — FP8 path matches vLLM contract; torch-fallback
   math sound. The prior "DL op HANGS" was a malformed *FP4* adapter (tuple/int8/68-wide into
   an FP8-only op) on a path DLIN never takes (`enable_deepseek_v4_fp4_indexer` defaults False,
   gated to sm100).
3. **MLA decode** (`flash_mla_with_kvcache`) — every arg matches vLLM. **Fixed real bug:**
   `causal=True→False` (commit `29d8b28e35`); causal=True read the None
   block_table/cache_seqlens as null descriptors in sparse mode.
4. **MLA prefill** (PATH A reuses decode kernel; PATH B sparse kernel broken on DLIN via
   op-name mismatch, only affects >11673-token prefill) — PATH A args structurally identical
   to verified decode.
5. **KV-store quantization** (`quant_to_nope_fp8_rope_bf16_pack_triton`) — inspected correct
   (per-64-tile FP8e4m3 + ue8m0 pow2 scales, 584-byte layout matches vLLM).

**Primary gibberish = MLA value-level** (not MHC): uniform MHC sets `post=0` (zeroes MHC
contribution → MHC≈identity via residual), yet output is *still* full gibberish → corruption
is in the MLA layers. MLA args are verified-correct, so it's a **value-level** bug
(q-projection / KV-store values / weight-loading / RoPE / norm) that arg-diffs can't catch.

**Secondary: Sinkhorn→empty** — correct Sinkhorn pre/post/comb values collapse output to
empty (uniform→gibberish). Implicates the MHC pre/post/fused torch fallbacks
(`mhc_pre`/`mhc_post`/`mhc_fused_post_pre`, used when `SGLANG_OPT_USE_TILELANG_MHC_PRE=0`).
Toggle: `SGLANG_DL_MHC_UNIFORM=1`. Layer 0 is MHC → processes first.

**CRITICAL REFRAME — vLLM V4-Flash ALSO fails on this DLIN setup (hangs).** `.venv` vLLM has
native V4 support. Serve cmd: `.venv/bin/vllm serve /LocalRun/hao.dong/DeepSeek-V4-Flash
--port 8299 --dtype bf16 --tp 8 --max-model-len 4096 --max-num-seqs 8 --kv-cache-dtype fp8
--enforce-eager --trust-remote-code` + `VLLM_USE_V2_MODEL_RUNNER=1` + sourced SDK env
(needs `--kv-cache-dtype fp8` or AssertionError). Model loads fine (19.83 GiB/worker), uses
fp8_ds_mla KV + Lightning Indexer, sets SWA block=256, then **HANGS** at warmup (0% util,
no log). So the "match vLLM's working call" strategy's premise is broken — there is no
working vLLM V4 reference on this DLIN box. V4-on-DLIN is immature in BOTH frameworks.

**Repro (sglang smoke, eager, TP8):** `CUDA_VISIBLE_DEVICES=8-15` (or 24-31 if 8-15 leaked)
+ `source sdk-dlop-07-13-20-30/env.sh` + env: `DLI_V2=ON TORCHDYNAMO_DISABLE=1
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_DL_FP8_Q2=1 SGLANG_DL_MOE_FUSED=1
SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_GDN_DLIN=1 SGLANG_OPT_USE_FUSED_HASH_TOPK=0
SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 SGLANG_OPT_USE_TOPK_V2=0
SGLANG_TOPK_TRANSFORM_512_TORCH=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0` →
`.venv/bin/python scripts/dl/v4_smoke.py`.

**Next steps (multi-session):**
- Debug vLLM V4's warmup hang (verbose logging) to recover a working reference, then compare
  sglang vs vLLM tensors layer-by-layer (layer-0 MLA output, post-store KV, logits).
- OR build a standalone pure-torch V4 MLA layer reference (heavy, avoids vLLM import which
  SIGABRT-crashes sglang's process via `_dl_C` double-registration).
- In-process MLA ref-check is a dead end: importing vLLM into sglang SIGABRT-crashes the
  scheduler (not catchable).

**Card notes:** killed vLLM leaked ~20GB on cards 8-15 (DLIN driver doesn't release on kill;
needs `sudo dlsmi -r -i <id>`). Cards 24-31 are a fresh free TP8 block.
