# DeepSeek-V4-Flash on DLIN: From Gibberish to CG — Debug Blog

**Date:** 2026-07-27
**Branch:** `support-dsv4`
**Model:** DeepSeek-V4-Flash (149GB, 43 layers, 256 experts top-6, MHC+MLA hybrid)

## TL;DR

V4-Flash went from **gibberish → correct output (1.6 tok/s eager)** → **CG captured (DL error 900 = 0, final MoE v3 integration in progress)**. Three correctness bugs fixed, 7 op types routed from sgl_kernel (CG-incompatible cuDNN reimpls) to native _dl_C (CG-capturable), MoE v3 identified as the last CG blocker. The fix is mechanical (match vLLM's `dl_invoke_fused_moe_v3` arg layout) but paused per user request.

---

## Part 1: Correctness — Three Bugs

### Bug #1: MLA decode `causal=True` (commit `29d8b28e35`)
The DL MLA decode op `flash_mla_with_kvcache` was called with `causal=True` in sparse mode. The op docstring says "causal... Only valid for dense attention." vLLM omits it (→False). Our sm120 and NVIDIA paths omit it too. Only the DL path had `True`.

**Fix:** `causal=True → causal=False`. Verified: "The capital of France is" → " Paris".

### Bug #2: `hc_post` no-op returned UNINITIALIZED memory (commit `2d1d8e4aa3`)
On DLIN, `_TilelangMissing._jit`'s `_raise` returns a `_noop` that does nothing. `mhc_post` allocated `out = torch.empty_like(residual)`, called the no-op `mhc_post_tilelang`, and returned `out` **unwritten**. `SGLANG_OPT_USE_TILELANG_MHC_POST` defaults True and was never set False on DLIN.

**Fix:** DL torch fallback in `mhc_post` matching the verified `hc_post_torch_impl` math. Layer 0 (MHC) residual no longer poisoned.

### Bug #3: MoE FP4 called as FP8 (commit `e5b26b5049`) — PRIMARY ROOT CAUSE
V4's MoE is mxfp4/FP4 (int8=2×FP4 packed). The fused op was called with `use_fp8_w8a8=True`, interpreting FP4-packed bytes as FP8 → 9000× residual explosion → gibberish. The op **supports FP4** via `use_mxfp4_w4a16` (confirmed in `common_extension_dl.cc:200`).

**Fix:** Switch quant flags to `use_mxfp4_w4a16=True` for FP4 models. Auto-detected via `self.is_fp4_expert` (works out-of-box, no env needed). Verified: "1+1=" → " 2", "Hello" → "I am fine".

**Investigation trail (wrong turns, for reference):**
- "MLA value-level bug" — WRONG (it was the MoE, verifiable via SKIP_MOE which didn't help because the indexer torch fallback also has syncs)
- "vLLM reference needed" — WRONG (vLLM V4 hangs at warmup too; the FP4 MoE crash was sglang's call-site bug)
- "FP4 kernel gap" — WRONG (the kernel existed; it was a one-flag call-site bug, same class as causal)

---

## Part 2: Performance — dlPTI Root Cause

### Throughput: 1.60 tok/s (TPOT 761ms, eager TP8)

**Per-component breakdown** (CUDA-event timing, `SGLANG_DL_TIMING=1`):
| Component | Time | % | Status |
|---|---|---|---|
| MLA projections (FP8 GEMV + rope + norm) | ~407ms | 53% | host overhead |
| MHC-mix + host | 157ms | 21% | torch fallback |
| MoE (FP4) | 37ms | 5% | fast |
| flash_mla (DL attention op) | 18ms | 2% | fast |

**dlPTI finding** (`dlpti_tools capture --activity-mask cu`):
- `cuLaunchKernel` = **1072ms** (~27,500 launches × 40µs host queue)
- `dlcuGraphExecNodeApplyPatch_` = **408ms** (per-step graph patching)
- **~50% of decode time is host launch overhead**, not GPU compute

**Conclusion:** DL ops (flash_mla, MoE) are fast (55ms total GPU). The bottleneck is 27,500 kernel launches/step serializing on the CPU → GPU starved. CG would fix this (1 replay vs 27,500 launches).

---

## Part 3: CG Investigation — The Full Story

### The problem
`cudaErrorStreamCaptureUnsupported` + `DL error, value:900` during CUDA graph capture. vLLM captures the same model successfully.

### The key insight (deep-dive agent finding)
vLLM's DL platform plugin **replaces ALL 5 CG-critical op types** with native `_dl_C` versions (CG-capturable). sglang had only replaced **0** of them — it was using `sgl_kernel.*` (cuDNN-descriptor reimplementations that are CG-incompatible).

| Op type | vLLM | sglang (before) | sgl_kernel problem |
|---|---|---|---|
| flash_mla | `_dl_C` | `sgl_kernel` | cuDNN reimpl, DL error 900 |
| indexer (mqa_logits) | `_dl_C` | `sgl_kernel` | cuDNN reimpl, DL error 900 |
| RMSNorm | `_dl_C` | `sgl_kernel` | cuDNN reimpl, DL error 900 |
| FP8 Linear | `_dl_C` | `sgl_kernel` | cuDNN reimpl, DL error 900 |
| MoE | `_dl_C.invoke_fused_moe_opt_v3` | `sgl_kernel.invoke_fused_moe_opt` | cuDNN descriptor, DL error 900 |

### Progress (DL error 900 count)
| Change | Errors remaining |
|---|---|
| Baseline (all sgl_kernel) | 12 |
| + flash_mla → _dl_C | 12 (flash_mla not in active decode path) |
| + indexer → _dl_C | 12 |
| + RMSNorm → _dl_C | 6 |
| + FP8 Linear (w8a8_matmul + Q2 gptq_dlblas_gemmex) → _dl_C | 2 |
| + SKIP_MOE | **0** ← MoE is the last blocker |
| + MoE v3 (invoke_fused_moe_opt_v3) | 0 (but hangs — needs real mabs dispatch) |

### The last blocker: MoE v3
`_dl_C.invoke_fused_moe_opt` (the "plain" version) is a **cuDNN descriptor API** — same class as sgl_kernel's version, CG-incompatible. vLLM uses `_dl_C.invoke_fused_moe_opt_v3` (**dlablas GEMM**, CG-capturable).

v3 has a **different signature** from the plain version:
- Plain: 22 args incl. 4 quant bools + trailing M
- v3: 18 args, replaces 4 bools with `weight_bits` int, drops trailing M

v3 also requires **real `moe_align_block_size` dispatch tensors** (vLLM always calls mabs before v3). sglang's trivial size-1 dispatch tensors cause v3 to hang.

### Next step (resume point)
1. Match v3 arg layout exactly to vLLM's `dl_invoke_fused_moe_v3` (`vllm-new-overlay/vllm/plugins/dl_platform_plugin/ops/_dl_ops.py:275-310`)
2. Always call `_mabs(_ti, _BM, num_experts)` before v3 (not just when `_need_mabs`)
3. Test: eager output correct + CG captures + TPOT measurement
4. Expected: TPOT 761ms → ~300-400ms (~2× speedup)

---

## Commits on support-dsv4

| Commit | Description |
|---|---|
| `29d8b28e35` | MLA causal=True→False |
| `2d1d8e4aa3` | hc_post no-op fix |
| `e5b26b5049` | MoE FP4 use_mxfp4_w4a16 (gibberish root cause) |
| `719879c564` | Per-layer hidden-state dump diagnostic tool |
| `055e35d24f` | CG prep: eliminate D2H syncs |
| `40148f7b5c` | flash_mla → _dl_C |
| `14af4d26a1` | indexer → _dl_C + metadata without deep_gemm |
| `6f7ae09106` | RMSNorm + FP8 Linear → _dl_C |
| `b63e5b10df` | MoE v3 + mabs (WIP, hangs) |

## Repro commands
```bash
# Eager (correct output):
CUDA_VISIBLE_DEVICES=24-31 source sdk-dlop-07-13-20-30/env.sh
export DLI_V2=ON TORCHDYNAMO_DISABLE=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_FP8_Q2=1 \
       SGLANG_DL_MOE_FUSED_MAX_M=2048 SGLANG_DL_GDN_DLIN=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 \
       SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 \
       SGLANG_OPT_USE_TOPK_V2=0 SGLANG_TOPK_TRANSFORM_512_TORCH=1
# FP4 MoE auto-detected via is_fp4_expert — no env needed
.venv/bin/python scripts/dl/v4_smoke.py

# CG test (fails until MoE v3 arg layout fixed):
# Same env + mem_fraction_static=0.75 + disable_cuda_graph=False + cuda_graph_max_bs_decode=4
```
