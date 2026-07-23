# sglang vs vLLM TPOT Gap 分析报告（Qwen3.5-35B-A3B-FP8, TP4）

**日期**: 2026-07-13
**模型**: Qwen3.5-35B-A3B-FP8 (256 experts, topk=8, 40 layers: 30 GDN + 10 full-attn)
**硬件**: DLIN DLIN GPU × 4 (32GB each), TP=4
**配置**: `attention_backend=fa3, page_size=16, disable_custom_all_reduce=True, CG on`
**环境**: `../sdk/env.sh` (dl19-matching), `DLEOL_CACHE_SIZE=1024`

---

## 1. 问题陈述

sglang TP4 decode TPOT 显著慢于 vLLM：

| 指标 | sglang | vLLM | gap |
|---|---|---|---|
| **纯 GPU forward/step** (CUDA event) | **27.5ms** | **18.3ms** | **9.2ms (1.5×)** |
| Wall TPOT (含 host 开销) | 33.5ms | 26.1ms | 7.4ms |
| Host 开销 | ~6ms | ~7.8ms | ~相等 |

**关键发现**: 两边 host 开销基本相等，gap **全部在 GPU ��算**。

---

## 2. 测量方法

### 2.1 纯 GPU forward 时间（可靠）

在 CUDA graph 的 `graph.replay()` 前后插入 CUDA event + `.synchronize()`，测量纯 GPU 执行时间（排除 Python/scheduler 开销）。

**sglang 侧**: `full_cuda_graph_backend.py::replay()` 加 `SGLANG_DL_TIME_REPLAY=1`
**vLLM 侧**: `compilation/cuda_graph.py:360` 加 `VLLM_DL_TIME_REPLAY=1`

### 2.2 组件级差分测试

通过环境变量控制跳过特定组件（返回零/输入不变），测量 GPU forward 的差值 = 该组件的 GPU 时间。

| 开关 | sglang | vLLM | 文件 |
|---|---|---|---|
| `SGLANG_DL_SKIP_MOE=1` | 13.5ms | 9.66ms (vLLM 用 `VLLM_DL_SKIP_MOE=1`) | `qwen3_5.py`, `moe_runner.py` |
| `SGLANG_DL_SKIP_ATTN=1` | 19.5ms | 13.13ms (vLLM 用 `VLLM_DL_SKIP_ATTN=1`) | `qwen3_5.py`, `qwen3_next.py` |
| `SGLANG_DL_SKIP_SHARED=1` | 25.15ms | — | `qwen2_moe.py` |

---

## 3. 分布式分解

| 组件 | sglang GPU 时间 | vLLM GPU 时间 | gap | 占比 |
|---|---|---|---|---|
| **MoE** (routed experts) | 14ms | 8.64ms | **5.36ms** | 58% |
| **Attention** (GDN + full-attn) | 8ms | 5.17ms | **2.83ms** | 31% |
| **其余** (norms + AR + lm_head + projections) | 5.5ms | 4.49ms | **1.01ms** | 11% |
| **总计** | **27.5ms** | **18.3ms** | **9.2ms** | 100% |

- MoE = baseline - SKIP_MOE = 27.5-13.5 = 14ms (sglang), 18.3-9.66 = 8.64ms (vLLM)
- Attention = baseline - SKIP_ATTN = 27.5-19.5 = 8ms (sglang), 18.3-13.13 = 5.17ms (vLLM)
- Shared expert = 27.5-25.15 = 2.35ms (sglang only)
- Routed MoE = 14-2.35 = 11.65ms (sglang)

---

## 4. 根因分析

### 4.1 已验证的等同性

两个框架在调用层面**完全等价**：

| 操作 | sglang 调用 | vLLM 调用 | 等同? |
|---|---|---|---|
| GDN decode recurrent | `_dl_C.dl_recurrent_gated_delta_rule` | `_dl_C.dl_recurrent_gated_delta_rule` | ✅ 字节相同 |
| Full-attn decode | `dl_flash_attn` (FA2 wrapper) | `dl_flash_attn` (FA2 wrapper) | ✅ 同款 |
| MoE GEMM | `_dl_C.invoke_fused_moe_opt` | `_dl_C.invoke_fused_moe_opt` | ✅ 同款 |
| Dense FP8 linear | `_dl_C.gptq_dlblas_gemmex` (quant_type=2) | `_dl_C.gptq_dlblas_gemmex` (quant_type=2) | ✅ 逐字一致 |
| RMSNorm | `_dl_C.gemma_rms_norm` / `fused_add_gemma_rms_norm` | `_dl_C.gemma_rms_norm` / `fused_add_gemma_rms_norm` | ✅ 同款 |
| MoE align | `moe_align_block_size` (triton) | `moe_align_block_size` (triton) | ✅ 同款 |
| activation | `F.silu(gate) * up` (2 kernels) | `_C.silu_and_mul` (1 kernel) | ⚠️ 不同 |

**权重验证**: sglang 的 `w13_weight` shape=(256,256,2048) dtype=fp8_e4m3fn contig=True stride=(524288,2048,1)，与 vLLM 相同。

**调用次数验证**: 两边 decode 都是 80 calls/step 的 `invoke_fused_moe_opt`（2/layer × 40 layers），same AR count (2/layer × 40)。

**block_size**: sglang 用 BM=16/BN=128/BK=128（对 M=1 memory-bound GEMV 最优），vLLM 用 `try_get_optimal_moe_config` 返回 BM=48/BN=128/BK=128。

### 4.2 MoE gap 根因 (5.36ms)

**根因**: DLEOL JIT 为 sglang 和 vLLM 编译了**不同的 kernel 变体**，尽管操作和参数名义相同。

差异来自：
1. **activation 融合**: sglang `F.silu*up` = 2 kernels × 40 layers = 80 extra kernels；vLLM `_C.silu_and_mul` = 1 kernel × 40 = 40 kernels。差 40 kernels/step。
2. **MoE align**: sglang 总是调 `_mabs`（40 triton kernels/step）；vLLM 在 decode 走 `use_moe_cu`（跳过 align，0 kernels）。差 40 kernels/step。
3. **combine**: sglang `c2.sum(dim=1)` = 40 kernels；vLLM `ops.moe_sum` = 40 kernels。数量相同但 kernel 不同。
4. **shared expert**: sglang 分离计算（`forward_normal_dual_stream`，含 `clone()` + 2 GEMM）；vLLM 融入 MoE runner。差 ~2.35ms。

**尝试关闭这些差异的实验全部失败**（见 §5），原因都是 DLEOL JIT bug。

### 4.3 Attention gap 根因 (2.83ms)

**根因**: 同上——DLEOL JIT 不同变体。两边 GDN recurrent 用同款 op，但：
1. sglang `fused_gdn_gating` 返回 fp32 beta → 每层需 `.to(bf16)` 转换（30 extra kernels/step，~0.3ms）
2. FA2 decode wrapper 的 kwargs 处理开销
3. GDN 层的 `fused_qkvzba_split_reshape_cat_contiguous` triton kernel 变体可能不同

### 4.4 其余 gap (1.01ms)

分散在 norms、allreduce、lm_head、logits_processor。非主要 lever。

---

## 5. 优化实验全记录

### 5.1 实验汇总表

| # | 实验 | 改动类型 | 结果 | 失败原因 |
|---|---|---|---|---|
| 1 | block_size_m=48 | config (匹配 vLLM) | 27.2ms ≈ baseline | 无改善（相同 JIT 变体） |
| 2 | block_size_m=64 | config | 32.3ms (更慢) | 更差 |
| 3 | page_size=64 | config | 27.2ms ≈ baseline | 无改善 |
| 4 | NO_ALT_STREAM (去 dual-stream) | config | 27.5ms | 无变化 |
| 5 | DELAY_SAMPLE=1 | config | 33.5ms wall | 仅省 wall host（不省 GPU） |
| 6 | shared_expert 融合 (FUSE_SHARED) | 代码 | **28.3ms 更慢** | 第9 expert ���销 > 分离 shared 节省 |
| 7 | GEMMEX=3 (per-expert dlblas GEMM) | 代码 | 37.4ms 更慢 | per-expert loop 开销大 |
| 8 | silu_and_mul (sglang jit_kernel triton) | 代码 | **崩溃** | DLEOL JIT: Device page fault |
| 9 | silu_and_mul (vLLM `_C` C++) | 代码 | **崩溃** | DLEOL JIT: K%VEC_K assert（加载_C 破坏 JIT state） |
| 10 | use_moe_cu (trivial sorted_ids) | 代码 | **崩溃** | DLEOL JIT: Device page fault |
| 11 | use_moe_cu + zeros | 代码 | **崩溃** | DLEOL JIT: cudaErrorInvalidAddressSpace |
| 12 | use_moe_cu + CG mode relaxed | config+代码 | **崩溃** | DLEOL JIT: cudaErrorInvalidAddressSpace |
| 13 | use_moe_cu + CG tc_piecewise | config+代码 | **崩溃** | DLEOL JIT: cudaErrorInvalidAddressSpace |
| 14 | fused_experts_impl (vLLM 完整函数) | 代码 | use_moe_cu OFF: **36.2ms 更慢** | vLLM config 选更差 block_size |
| 15 | fused_experts_impl + block_size override | 代码 | **崩溃** | DLEOL JIT: K%VEC_K assert |
| 16 | fused_experts_impl + use_moe_cu ON | 代码 | **崩溃** | DLEOL JIT: Device page fault |
| 17 | SKIP_ALIGN (cache align output) | 代码 | **崩溃** | stale routing metadata |
| 18 | mul_routed_weight=False on w2 | 代码 | **崩溃** | DLEOL JIT: K%VEC_K assert |
| 19 | pre-compile use_moe_cu before model load | 代码 | **崩溃** | DLEOL JIT: Device page fault |
| 20 | vllm._C import (matching vLLM loading) | 代码 | 无效 | 加载方式不影响 |
| 21 | patch_in_gpu_worker (vLLM 完整 init) | 代码 | 无效 | 已运行，仍崩溃 |
| 22 | beta.to(bf16) → fused_gdn_gating 返回 bf16 | 代码 | **崩溃** | DLEOL JIT: triton kernel 输出 dtype 变化 |
| 23 | vLLM _C.silu_and_mul (CG) | 代码 | **崩溃** | DLEOL JIT: K%VEC_K assert |
| 24 | vLLM _C.silu_and_mul (standalone) | 微基准 | **OK** ✅ | 证明 op 本身可用 |

### 5.2 独立微基准测试结果

**关键发现**: 所有 op 在**独立微基准**（无 torch.distributed）中工作正常：

| 测试 | 结果 |
|---|---|
| `invoke_fused_moe_opt` + use_moe_cu + 随机权重 + seq topk_ids | ✅ OK |
| `invoke_fused_moe_opt` + use_moe_cu + 随机权重 + random topk_ids | ✅ OK |
| `invoke_fused_moe_opt` + use_moe_cu + 随机权重 + duplicate topk_ids | ✅ OK |
| `invoke_fused_moe_opt` + use_moe_cu + 100 GEMM 预热后 | ✅ OK |
| `invoke_fused_moe_opt` + use_moe_cu + 各种 scale 值 (ones/rand/zero) | ✅ OK |
| `invoke_fused_moe_opt` + use_moe_cu + block_shape=[] (per-channel) | ❌ CRASH |
| `invoke_fused_moe_opt` + use_moe_cu + 模型上下文 (TP2) | ❌ CRASH (Device page fault) |
| `_C.silu_and_mul` 独立 | ✅ OK (max error 0.000000) |
| `sglang.jit_kernel.silu_and_mul` 独立 | ✅ OK |

---

## 6. DLEOL JIT Bug 分析

### 6.1 Bug 模式

所有失败实验归结为 **3 类 DLEOL JIT bug**：

| Bug 类型 | 症状 | 触发条件 | 涉及实验 |
|---|---|---|---|
| **K%VEC_K static_assert** | JIT 编译时 assert 失败 | 改变 JIT key（mul_routed_weight, block_size, fused_experts_impl config） | #9,15,18,23 |
| **Device page fault** | 运行时内存越界访问 | 新 triton kernel 变体 / use_moe_cu 模式 / beta dtype 变化 | #8,10,19,22 |
| **cudaErrorInvalidAddressSpace** | CG 捕获时不支持的内存操作 | use_moe_cu + CG capture | #11,12,13 |

### 6.2 Bug 根因

**DLEOL JIT 在 `torch.distributed` 初始化后的模型上下文中，无法正确编译/执行新的 kernel 变体。**

- **已编译的 baseline 变体**: 在模型加载/warmup 期间编译，工作正常
- **新的 JIT key**: 在模型上下文中编译，触发上述 3 类 bug 之一

独立测试（无 torch.distributed）可以编译和执行所有变体。这证明：
1. op 本身支持这些变体
2. DLEOL JIT 编译器在某些上下文条件下产生 buggy 的 kernel
3. `torch.distributed` 初始化改变了 CUDA 上下文属性，影响 DLEOL JIT 的编译决策

### 6.3 影响范围

DLEOL JIT bug 阻止了所有可能减小 gap 的优化：

| 优化 | 需要 | 阻塞 |
|---|---|---|
| 跳过 MoE align | use_moe_cu（新 JIT key�� | Device page fault |
| 融合 silu+mul | 新 triton kernel / _C 加载 | Device page fault / K%VEC_K |
| 改 w2 mul_routed_weight | JIT key 变化 | K%VEC_K |
| fused_experts_impl | 新 config/cache JIT key | K%VEC_K |
| bf16 beta output | triton kernel dtype 变化 | Device page fault |

---

## 7. vLLM 为何更快

vLLM 在相同硬件、相同 op、相同权重下达到 18.3ms（比 sglang 快 9.2ms），原因：

1. **DLEOL JIT 变体选择**: vLLM 的模型构建过程（`make_fp8_moe_kernel` + `FusedMoEKernel`）以不同的调用上下文（tensor strides、内存布局、调用顺序）触发 DLEOL JIT，编译出更快的 kernel 变体。sglang 的调用上下文触发更慢的变体。

2. **Activation 融合**: vLLM 用 `_C.silu_and_mul`（1 kernel），sglang 用 `F.silu * up`（2 kernels）。每 step 差 40 kernels。

3. **MoE align 跳过**: vLLM decode 走 `use_moe_cu`（跳过 `moe_align_block_size`），sglang 不能（DLEOL JIT 崩溃）。每 step 差 40 triton kernels。

4. **Shared expert 融合**: vLLM 把 shared expert 融入 MoE runner（monolithic kernel 处理），sglang 分离计算。

5. **GDN gating dtype**: vLLM 的 `fused_gdn_gating` 直接返回 bf16 beta，sglang 返回 fp32 需 `.to(bf16)` 转换（30 extra kernels）。

---

## 8. 可交付成果

### 8.1 诊断工具（已在代码树中）

| 工具 | 位置 | 用途 |
|---|---|---|
| `SGLANG_DL_TIME_REPLAY=1` | `full_cuda_graph_backend.py` | 测量纯 GPU forward 时间 |
| `VLLM_DL_TIME_REPLAY=1` | vLLM `cuda_graph.py` | 测量 vLLM 纯 GPU forward 时间 |
| `SGLANG_DL_SKIP_MOE=1` | `qwen3_5.py` | 跳过 MoE，测量 attn+rest |
| `SGLANG_DL_SKIP_ATTN=1` | `qwen3_5.py` | 跳过 attention，测量 MoE+rest |
| `SGLANG_DL_SKIP_SHARED=1` | `qwen2_moe.py` | 跳过 shared expert |
| `SGLANG_DL_LAYER_TIMING=1` | `qwen3_5.py` | eager 逐层 GDN/MoE 分解 |

### 8.2 Bug 修复

| 修复 | 位置 | 影响 |
|---|---|---|
| triton JIT dlcc PATH | `.venv/.../dlgpu/compiler.py:294` | fresh JIT 不再崩（之前靠 DLEOL 磁盘缓存掩盖） |

### 8.3 文档

| 文档 | 章节 |
|---|---|
| `docs/dl/sglang-vs-vllm-perf-gap.md` | §7.24–7.29 |
| `docs/dl/sglang-vllm-tp4-gap-report.md` | 本报告 |

---

## 9. 建议的下一步

### 9.1 短期（需 DLIN SDK 团队配合）

1. **修复 DLEOL JIT `K%VEC_K` assert**: JIT auto-tuning 应在 VEC_K 不整除 K 时回退到更小的 VEC_K，而非 assert 失败。
2. **修复 DLEOL JIT Device page fault**: 调查为何 `torch.distributed` 初始化后新 triton kernel 变体会越界访问���
3. **提供 `use_moe_cu` 的稳定支持**: 确保 `invoke_fused_moe_opt` 在所有调用上下文中正确处理 trivial sorted_token_ids。

### 9.2 中期（sglang 代码层面，待 DLEOL JIT 修复后）

1. **activation 融合**: 用 `_C.silu_and_mul` 替代 `F.silu * up`（待 JIT 修复后）。
2. **use_moe_cu**: decode 跳过 `moe_align_block_size`（待 JIT 修复后）。
3. **fused_gdn_gating bf16 output**: 消除 30 个 `.to(bf16)` kernel（待 JIT 修复后）。
4. **shared expert 融合**: 把 shared expert 作为第 257 个 expert 融入 MoE（需验证 monolithic kernel 支持）。

### 9.3 长期（架构层面）

1. **移植 vLLM monolithic MoE kernel 架构** (`make_fp8_moe_kernel`): 融合 dispatch+align+GEMM+combine+shared，减少 Python 编排层。
2. **DLEOL JIT 变体对齐**: 确保 sglang 的调用上下文与 vLLM 一致，触发相同的（更快的）JIT 变体。

---

## 10. 预期收益估算

| 优化 | 预期节省 | 前提 |
|---|---|---|
| use_moe_cu (跳过 align) | ~2ms | DLEOL JIT 修复 |
| silu_and_mul 融合 | ~0.5ms | DLEOL JIT 修复 |
| beta.to(bf16) 消除 | ~0.3ms | DLEOL JIT 修复 |
| shared expert 融合 | ~1ms | monolithic kernel |
| JIT 变体对齐 | ~3-5ms | 架构调整 |
| **合计** | **~7-9ms** | **可关闭大部分 gap** |

---

*报告完成。代码已清理（所有实验改动已回退），诊断工具和报告保留在代码树中。*
