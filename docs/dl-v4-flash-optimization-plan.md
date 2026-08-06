# DeepSeek-V4-Flash sglang 优化方案

> 基于 V4-Flash + DLIN KS38 32GB×8 TP8 的静态代码分析 + 实测 profiling。
> 当前最优：**sglang EAGLE 15.83 tok/s**（+85% over 8.54 基线，比 vLLM 14.76 快 7%）。
> 目标：**20 tok/s 单流**。

## 1. 性能现状

### 1.1 CG TPOT 分解（ctx=512, 实测）

| 组件 | CG 时间 | 占比 | 性质 |
|---|---|---|---|
| MoE (cuDNN FP4 + comm) | 21.8ms | 28% | GPU compute + TP8 reduce-scatter |
| attention (flash_mla + indexer) | 39.6ms | 52% | flash_mla 2.4ms + indexer 10.3ms + proj 5.7ms + 其他 |
| 其他 (norm/sample/hc/rope/KV) | 15.2ms | 20% | elementwise + IPC |
| **总计** | **76.6ms** | **100%** | |

### 1.2 20 tps 的数学条件

```
20 tok/s (50ms/token) with EAGLE steps=1 (accept_length=2):
  EAGLE step = draft(~5ms) + verify(~121ms) = ~126ms → 63ms/token
  需 step ≤ 100ms → verify ≤ 95ms → base_M1 ≤ 58.7ms
  当前 base_M1 = 76.6ms，需砍 17.9ms
```

### 1.3 已完成的优化（commit `2ea25190a`）

- ✅ indexer Triton kernel（DL op 262208-entry padding → valid-only，+18%）
- ✅ EAGLE steps=1（100% accept，+21%）
- ✅ ctx=512 + maybe_flush CG 修复（+28%）
- ✅ MHC Triton port（vLLM → sglang，3 bug fixes）
- ✅ vLLM dl_mhc_triton.py static_range 修复（让 vLLM 能跑）

---

## 2. 优化方案（按优先级 + 预期收益排序）

### P0: 快速修复（<10 行/项，立即实施）

| # | 优化 | 文件 | 预期收益 | 难度 |
|---|---|---|---|---|
| 1 | **缓存 MoE flags** — `_use_reduce_scatterv`/`_use_reduce_scatter` 等 env var 检查移到 `__init__` | deepseek_v4.py:1790-1812 | ~0.3ms | trivial |
| 2 | **GEMMEX=2 批量化** — 默认 `SGLANG_DL_MOE_GEMMEX=2`（2 GEMMs vs 16 per-expert loops） | fp8.py:2280-2298 | ~0.5-1ms | env var |
| 3 | **共享专家本地计算** — `SGLANG_DP_SHARED_EXPERT_LOCAL=1` 默认开启 | deepseek_v4.py:1823 | ~2-3ms（MoE 15-25% FLOPs） | env var |
| 4 | **match_num_queries 批量化** — 将 5 次 match 调用合并为 1 次 | indexer.py:728-733, backend.py:1676-1679 | ~0.1ms | trivial |
| 5 | **debug assert 移出热路径** — `% 64 == 0` 等检查移到 init | backend.py:1693-1698 | ~0.05ms | trivial |
| 6 | **cache buffer views 预计算** — shape 不变时缓存 view | backend.py:1634-1665 | ~0.2ms | trivial |
| 7 | **EAGLE draft loop: 缓存 out_cache_loc** — 预计算 per-step slices | eagle_worker_v2.py:647 | ~0.2ms/step | trivial |
| 8 | **EAGLE: 跳过 prefix_tail when last_page==0** | eagle_worker_common.py:283 | ~0.3ms/batch | trivial |

**P0 合计预期：~3.5-5ms → 76.6→72ms → 13.9 tok/s → EAGLE 16.8 tok/s**

---

### P1: 中等工作量（需要新 kernel 或架构调整）

| # | 优化 | 文件 | 预期收益 | 难度 |
|---|---|---|---|---|
| 9 | **indexer compressor Triton 融合** — fuse compressor + weights + Q prep 成 1 个 kernel | indexer.py:604-625 | ~2-3ms | medium |
| 10 | **indexer logits + topk 融合** — 避免物化完整 logits tensor | indexer.py:766-827 | ~1-2ms | medium |
| 11 | **MLA projection 融合** — fuse wq_b/wkv + norm + RoPE 成 1 个 kernel | deepseek_v4.py:664-706 | ~0.5ms | medium |
| 12 | **FP8 scale 全路径缓存** — 非 SGLANG_DL_MOE_FUSED 路径也缓存 contiguous scales | fp8.py:2270-2272 | ~2-3ms | easy |
| 13 | **attention metadata 缓存** — seq_lens_casual 等 per-forward 而非 per-layer | backend.py:681-706 | ~0.2ms | medium |
| 14 | **EAGLE verify buffer 预分配** — tree_mask_buf / window widened tensors 复用 | eagle_worker_common.py:136-145,346 | ~0.3ms/verify | medium |
| 15 | **EAGLE grammar mask 异步生成** — CPU mask 与 GPU verify 并行 | eagle_worker_common.py:531-537 | ~0.5-1ms | medium |
| 16 | **scheduler recv_requests 异步化** — ZMQ recv 与 GPU 并行 | scheduler.py:1577-1594 | ~0.5-2ms | hard |

**P1 合计预期（叠加 P0 后）：~7-12ms → 72→60-65ms → 15.4-16.7 tok/s → EAGLE 18.6-20.2 tok/s**

---

### P2: 架构级改动（需深入模型/框架代码）

| # | 优化 | 描述 | 预期收益 | 难度 |
|---|---|---|---|---|
| 17 | **实现 V4 Frozen-KV MTP** — 在 deepseek_v4.py 实现 `bind_frozen_kv_context` + `build_frozen_kv_mtp_context`，draft 直接复用 target 的 KV cache（不分配独立 draft KV） | 5-13ms/verify（避免 draft KV 管理开销） | hard（2-3 天） |
| 18 | **实现 V4 NextN draft-only forward** — `is_nextn=True` 时跳过 compressor/indexer topk（draft 只需相对预测），复用 target attention pattern | 2-5ms/draft_step | hard |
| 19 | **sglang in-process TP mode** — 消除 scheduler↔tokenizer ZMQ IPC（8.8ms/step），改用单进程 | 8.8ms | very hard（架构改动） |
| 20 | **修复 DLIN CG stream capture for MTP** — 让 vLLM/sglang 的 MTP spec 能用 CG（当前 cudaErrorStreamCaptureInvalidated） | 解锁 MTP spec → accept 可能更高 | hard（DLIN SDK bug） |

**P2 合计预期：如果 #17+#18 成功 → EAGLE multiplier 从 1.21→1.5+ → 20+ tok/s**

---

## 3. 关键路径分析

### 3.1 Decode 单步前向流程（43 层 × 每层）

```
input_norm → hc_pre → [multi-stream parallel: indexer | KV-proj | compressor]
→ flash_mla → o_proj → hc_post → post_norm → MoE (gate+w1+act+w2+reduce_scatter)
→ all-reduce → residual add → output
```

### 3.2 EAGLE spec 单步流程

```
draft_forward (mtp.0 layer, 1 autoregressive step):
  embed → full V4 layer forward (norm + attn + MoE) → sample → draft_token

verify (target model, batch=1+num_draft):
  full 43-layer V4 forward with [bonus + draft_tokens]
  → compare draft logits vs target logits → accept/reject
  → produce accepted_tokens + new_bonus
```

### 3.3 瓶颈位置（GPU 99% util，无 idle）

| 位置 | 时间 | 根因 |
|---|---|---|
| MoE GEMM (w1+w2) | ~14ms | cuDNN FP4 grouped GEMM，GPU compute-bound |
| MoE reduce-scatter | ~8ms | TP8 all-to-all，comm-bound |
| flash_mla sparse | 2.4ms | vendored dldnn MLA kernel |
| indexer logits | 0.15ms | ✅ 已优化（Triton valid-only kernel） |
| indexer compressor | ~3ms | torch ���现，unfused |
| indexer topk | ~2ms | torch topk_transform（native crash on DLIN） |
| MLA projections | ~6ms | wqkv_a + wq_b + wkv + wo_a + wo_b，separate kernels |
| hc_pre/post | ~6ms | Triton ported from vLLM |
| norm/rope/KV-store | ~5ms | partially fused (fused_qk_norm_rope_swa_store) |

---

## 4. 实施路线图

### 阶段 1：快速修复（1-2 天）→ 目标 16.8 tok/s

1. 实施 P0 #1-#8（全部 <10 行改动）
2. 逐项验证正确性（WARMUP "Paris" + 200 token decode）
3. 逐项测量 TPOT

### 阶段 2：融合 kernel（3-5 天）→ 目标 18-20 tok/s

1. P1 #12（FP8 scale 缓存，最 easy 的 2-3ms）
2. P1 #9（compressor Triton 融合）
3. P1 #10（logits + topk 融合）
4. P1 #11（MLA projection 融合）
5. P1 #16（scheduler 异步化）

### 阶段 3：架构突破（1-2 周）→ 目标 20+ tok/s

1. P2 #17（Frozen-KV MTP — 解锁更高 accept_rate）
2. P2 #18（NextN draft-only — 减小 draft forward 成本）
3. P2 #20（DLIN CG stream capture 修复 — 解锁 MTP CG）

---

## 5. 实测对比基线

| 配置 | tok/s | TPOT | 来源 |
|---|---|---|---|
| 原始基线 (CG, default ctx) | 8.54 | 117ms | 实测 |
| + indexer Triton kernel | 10.16 | 98ms | 实测 |
| + ctx=512 (maybe_flush fix) | 13.05 | 76.6ms | 实测 |
| + EAGLE steps=1 | **15.83** | **63.2ms** | 实测 |
| vLLM non-spec CG | 14.76 | 67.8ms | 实测 |
| vLLM MTP eager | 8.26 | 121ms | 实测 |
| vLLM MTP CG | ❌ fail | — | DLIN stream capture |
| 20 tps 目标 | 20.0 | 50ms | — |

---

## 6. 关键约束

1. **V4 只有 1 个 mtp layer** → EAGLE accept_length 上限 = 2.0
2. **V4 asserts topk=1** → 不支持 tree speculation
3. **DLIN CG stream capture** → MTP spec 不能用 CG（cudaErrorStreamCaptureInvalidated）
4. **DLIN native topk crash** → kPDL/cooperative_groups 是 NVIDIA-only
5. **torch.compile on DLIN hurts** → combo_kernels fused code < manual fused（12.89 < 13.05）
6. **GPU 99% util** → 无 Python idle，gap 纯 GPU compute

---

## 附录 A: 所有代码位置

### 已优化（commit `2ea25190a`）
- `jit_kernel/dsv4/dl_mhc_triton.py` — MHC pre/post Triton kernels（ported from vLLM）
- `jit_kernel/dsv4/dl_mqa_logits_triton.py` — indexer valid-only MQA logits kernel
- `layers/quantization/dl_moe_profile.py` — deferred-sync profiler（maybe_flush CG fix）
- `models/deepseek_v4.py` — DLIN paths: hc_pre/post, multi-stream, skip flags
- `layers/attention/dsv4/indexer.py` — Triton kernel dispatch + torch ref gate
- `layers/attention/dsa_backend.py` — deep_gemm import guard

### 待优化（本文档）
- `models/deepseek_v4.py:1790-1812` — MoE flags caching
- `layers/quantization/fp8.py:2280-2298` — GEMMEX batching
- `models/deepseek_v4.py:1823` — shared expert local
- `layers/attention/dsv4/indexer.py:604-625` — compressor fusion
- `speculative/eagle_worker_v2.py:647` — draft loop optimization
- `speculative/frozen_kv_mtp_worker_v2.py` — Frozen-KV MTP (需 V4 实现)
- `managers/scheduler.py:1577-1594` — async recv_requests

### 实测脚本
- `scripts/dl/v4_decode_profile.py` — CG/eager toggle, spec, profiler
- `scripts/dl/v4_vllm_v2_bench.py` — vLLM V2 bench
- `scripts/dl/v4_vllm_mtp_bench.py` — vLLM MTP bench
- `scripts/dl/v4_mqa_logits_microbench.py` — indexer kernel isolate bench

---

## 7. P0 实测结果（2026-08-06）

### P0 #2: GEMMEX=2 default → ❌ REVERTED
- **结果**: EAGLE tok/s 从 15.83 降到 14.08（**-11% 回归**）
- **原因**: GEMMEX=2 的 Python-side weight gather（`layer.w13_weight[_ti1d].reshape(...)` + `.contiguous()`）比默认的 `invoke_fused_moe_opt`（DLEOL cache + cu kernel dispatch）慢。Python gather 产生额外内存拷贝。
- **结论**: 默认 fused path 已经是最优。GEMMEX 是探索性路径，不适合作为默认。
- **修复**: 已 revert，保留 `_dl_moe_gemmex` 变量名简化（行为不变）。

### P0 #3: Shared expert local default ON → ⚠️ 无效果
- **结果**: 无性能变化（`_use_tp_moe_gather` 在 TP8 无 DP attention 时为 False）
- **原因**: `_do_shared_local` 需要 `_use_tp_moe_gather=True`，而 `attn_dp_size > 1` 在纯 TP8（无 DP attention）时为 False。
- **结论**: 保留 default ON（对有 DP attention 的场景有益，对当前配置无害）。

### P0 #1, #4-#8: 未实施
- GPU 内存泄漏问题（多次实验后 GPU 内存未完全释放）导致无法继续测试。
- 需要物理重启 GPU 或等待内存释放。

### 教训
1. **先测后改**: GEMMEX=2 在理论分析上应该更快（2 GEMM vs 16），但实测更慢。Python gather 开销 > kernel launch 节省。静态分析必须配实测验证。
2. **flag 依赖链**: shared expert local 依赖 `_use_tp_moe_gather`，而后者依赖 DP attention 配置。默认值改了但执行路径没变。
3. **GPU 内存管理**: DLIN KS38 在多次实验后内存不完全释放，需要谨慎管理实验顺序。
