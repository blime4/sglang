# SGLang on DLIN: 投机解码、Kernel 解耦与推理优化

SGLang Team2026 年 7 月 22 日

过去一周，我们在 DLIN（DLIN GPU）推理栈的三个方向上取得了重要进展：**基于 DFlash 和 MTP 的投机解码**、**Kernel 层对 vLLM 的完全解耦**、以及 **torch.compile Phase II 融合编译**。我们还搭建了一套一键式基准测试工具，在相同硬件上对 SGLang 和 vLLM 进行了可复现的对比。

- **DFlash on DLIN**：在 Qwen3.5-35B 上通过端到端正确性验证，`dl_chunk` prefill 实现验证正确，prefill 吞吐提升约 **+5%**。
- **MTP（Qwen3.5）**：投机解码通路在 DLIN 上完整跑通，`FROZEN_KV_MTP` CUDA graph 修复实现稳定多步投机。
- **Kernel 解耦 Phase 1–4**：所有 DLIN 专用 kernel——rotary embedding、fused MoE、marlin GEMM、gemma RMS norm——从 `vllm._custom_ops` / `vllm._dl_C` 迁移至 `sgl-kernel`，消除上游编译依赖。
- **torch.compile Phase II Stage 0+1**：`PostGradPassManager` 融合框架上线，包含 activation quantization fusion、pattern matcher 基础设施，以及双策略 compile + CUDA graph 调度。
- **SGLang 自研 Flash Attention**：自定义 `_sgl_fa2_C` kernel 消除与 vLLM flash-attn 的 namespace 冲突，decode 吞吐稳定在 **35.0 tok/s**。
- **Decode 时延分解**：Host 端时序分析将剩余 **6.2ms/token** 的差距精确定位到 ZMQ IPC（scheduler ↔ tokenizer 进程间通信）。
- **SGLang vs. vLLM 基准测试**：RadixAttention 在前缀共享场景下实现 **1.6–2.3×** 吞吐领先，冷→暖 SC1 加速比达 **16.4×**。
- **完整文档体系**：7 篇技术博客（中英文）、3 幅架构图、kernel 移植规划与 torch.compile 路线图设计文档。

---

## 1. 投机解码：DFlash 与 MTP

### DFlash：在 DLIN 上通过正确性验证

我们在 DLIN 上实现了 DFlash 投机解码并完成验证。工作涵盖正确性测试 harness、自定义 prefill kernel、以及与 SGLang 投机解码运行时的集成。

正确性 harness（`scripts/dl/dflash_correctness.py`）执行完整的 DFlash 流水线——draft 生成、目标验证、token 接受——并将输出分布与非投机基线对比。Qwen3.5-35B 上的端到端正确性已通过验证。

我们还引入了 `dl_chunk`——一种为 DFlash draft 路径设计的自定义 chunked prefill kernel，将 draft 模型在多个投机位置上的前向计算合并为一次执行。相比朴素的顺序 draft 推理路径，整体 prefill 吞吐提升约 **+5%**。

**复现正确性验证：**

```bash
python3 scripts/dl/dflash_correctness.py \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --tp 4
```

### MTP：在 DLIN 上跑通端到端

Qwen3.5 的多 token 预测支持（MTP，`python/sglang/srt/models/qwen3_5_mtp.py`）在 DLIN 上达到端到端可用。修复集中在 `FROZEN_KV_MTP` CUDA graph runner（`frozen_kv_mtp_cuda_graph_runner.py`）：

- 使用 frozen KV 时 MTP draft head 的正确 graph capture
- 跨 replay 迭代的 tensor 地址稳定性
- 可变长度 draft 输出在 graph 边界内的正确处理

**NGRAM / DFlash / MTP 三维加速矩阵**已完成量化，为后续动态调度器的工作提供了数据基础。

---

## 2. Kernel 解耦：Phase 1–4

DLIN GPU 支持长期依赖 vLLM 的 `_custom_ops`（`_dl_C` 扩展）共享 kernel 实现。这造成了一个双向耦合：每次 kernel 改动都需要两个仓库的协同修改。过去一周我们系统性地切断了这一依赖。

| Phase | 范围 | 状态 |
|-------|------|------|
| Phase 1 | rotary embedding + fused MoE：从 `vllm._custom_ops` 切换至本地路径 | 已合并 |
| Phase 2 | marlin GEMM：本地 shim，移除 `vllm._custom_ops` 导入 | 已合并 |
| Phase 3 | `suppress_other_loggers`：移除 `vllm.logger` 依赖 | 已合并 |
| Phase 4a | `gemma_rms_norm`：在 `sgl-kernel` DL 构建中编写并注册 CUDA kernel | 已合并 |
| Phase 4e | layernorm GEMMA 路由：完全切换到 `sgl_kernel`，`vllm._dl_C` 不再使用 | 已合并 |

`gemma_rms_norm` kernel（`sgl-kernel/csrc/elementwise/gemma_rmsnorm_dl.cu`，223 行）通过 DL 扩展构建（`sgl-kernel/csrc/common_extension_dl.cc`）注册为 `torch.ops.sgl_kernel.gemma_rms_norm`。该 kernel 处理 Gemma 系列模型在 DLIN 硬件上所需的融合归一化+缩放操作，使用面向 DLIN 存储子系统的向量化内存访问。

完整移植规划见 `docs/dl/sglang-port-vllm-kernels-plan.md`（476 行）。

---

## 3. torch.compile Phase II：PostGradPassManager 与算子融合

Phase II 在 DLIN 上引入了基于 `PostGradPassManager` 架构的 Stage 0 和 Stage 1 融合能力。

### 融合 Pass

我们在 `python/sglang/srt/compilation/passes/fusion/` 中实现了三个新的编译 pass：

- **`dl_act_quant_fusion.py`**（212 行）：将 activation quantization op 融合到前驱计算 op 中，减少全局内存往返。该 pass 匹配 `silu_and_mul → quant` 和 `rms_norm → quant` 子图，将其重写为融合实现。
- **`dl_matcher_utils.py`**（199 行）：跨所有 fusion pass 共用的图模式匹配工具——子图提取、op 谓词检查、tensor meta 校验。
- **`dl_pattern_matcher.py`**（181 行）：通用 FX 图模式匹配引擎，支持声明式融合规则定义。

### 双策略 Compile + CUDA Graph

小 batch 受益于 torch.compile 的算子融合但受 triton kernel 编译开销影响。大 batch 更适合 CUDA graph capture。双策略（`python/sglang/srt/compilation/torch_compile_decoration.py`）根据 batch size 自动选择执行路径：

```bash
# 小 batch 走 compile 融合，大 batch 走 CUDA graph
--enable-torch-compile --enable-dl-act-quant-fusion
```

单元测试（`tests/dl/test_compile_fusion.py`，185 行）验证每个 fusion pass 相对于 eager 模式执行的正确性。

---

## 4. SGLang 自研 Flash Attention

DLIN 的 flash attention 路径此前依赖 `vllm_flash_attn`（PyPI 包），该包将 CUDA kernel 注册在 `_vllm_fa2_C` torch library namespace 下。当 SGLang 在同一进程中加载 vLLM 的 fused MoE kernel（`SGLANG_DL_MOE_VLLM=1`）时，两个 `.so` 竞争同一 namespace，导致 `c10::Error` 异常终止。

修复方案是编译一个 SGLang 自有的 `flash_attn_2_cuda.so`（使用独立 namespace `_sgl_fa2_C`）并提供 Python wrapper（`python/sglang/srt/layers/attention/sgl_flash_attn.py`）。attention 后端（`dl_flash_attn.py`）现在从 SGLang 的 wrapper 而非 vLLM 导入：

```
之前: dl_flash_attn.py → import vllm_flash_attn → torch.ops._vllm_fa2_C
之后: dl_flash_attn.py → import sgl_flash_attn → torch.ops._sgl_fa2_C
```

**Flash attention 零 vLLM 依赖。** 变更后吞吐稳定在 **35.0 tok/s**，无退化。

### Decode 时延分解

Host 端时序分解揭示了 decode step 的耗时构成：

| 组件 | SGLang | vLLM | 说明 |
|-----------|--------|------|------|
| GPU kernel time | 20.7 ms | 20.7 ms | 完全相同——同一套 kernel |
| Host 调度 + IPC | ~9 ms | ~5.7 ms | 差距来源 |
| Decode step 总计 | ~29.7 ms | ~26.4 ms | |
| 每 token 差距 | | **6.2 ms/token** | 100% 来自 ZMQ IPC（scheduler ↔ tokenizer） |

这一差距是架构性的：SGLang 使用独立的 tokenizer 进程并通过 ZMQ 通信，而 vLLM 将 scheduler 和 tokenizer 放在同一进程中。详细的 profiling 报告见 `docs/dl/sglang-dlin-decode-gap-profiling-blog.md`。

**Profiling 命令：**

```bash
python3 scripts/dl/profile_decode_step.py \
  --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --tp 4 --output ./decode_trace.json
```

---

## 5. SGLang vs. vLLM：数据驱动的对比

我们构建了可复现的基准测试工具（`scripts/dl/showcase_prefix_sharing.py`）和一键对比脚本（`run_sglang.sh compare`）。两个引擎在相同硬件（DLIN KS38, GPU 0–3, TP4）上使用相同模型（Qwen3.5-35B-A3B-FP8）、相同 prompt、温度 0、best-of-3 运行。

### 结果

| 场景 | SGLang | vLLM | 优势 |
|----------|--------|------|------|
| SC1 前缀共享 — warm（缓存命中） | **5.56 s** | 13.0 s | SGLang **2.3×** |
| SC1 cold → warm 加速比 | **16.4×** | 1.0× | SGLang **16.4×**（RadixAttention） |
| SC2 多轮对话 — 5 轮平均 | **8.74 s** | 14.4 s | SGLang **1.6×** |
| SC3 并发批 — 吞吐 | **6.0 tok/s** | 2.6 tok/s | SGLang **2.3×** |
| SC4 结构化 JSON（短文本，预热后） | 31.7 tok/s | 34.6 tok/s | vLLM +9%（无前缀复用） |
| 纯 decode TPOT（无前缀） | 35.0 tok/s | 37.9 tok/s | vLLM +8% |

SGLang 在**所有涉及前缀复用、多轮对话或并发批处理的场景**中全面领先——而这覆盖了大多数真实服务场景。vLLM 仅在无状态单请求场景下有微弱优势。vLLM 的 Automatic Prefix Caching（APC）在该模型上因硬 assert 失败无法开启。

**一键对比：**

```bash
bash run_sglang.sh compare \
  --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --tp 4
```

完整报告（8 页）见 `docs/dl/sglang-vs-vllm-showcase-dlin.md`。

---

## 6. 基准测试基础设施

本周新增了多个脚本和工具以支持可复现的基准测试：

| 脚本 | 用途 |
|--------|--------|
| `scripts/dl/showcase_prefix_sharing.py` | 多场景 SGLang vs. vLLM 基准测试 |
| `scripts/dl/dflash_correctness.py` | DFlash 输出正确性验证 |
| `scripts/dl/dflash_repro.py` | DFlash 端到端复现 |
| `scripts/dl/dl_chunk_test.py` | Chunked prefill 正确性与延迟 |
| `scripts/dl/prefill_latency_test.py` | 独立 prefill 延迟测量 |
| `scripts/dl/mtp_correctness.py` | MTP 输出正确性 |
| `scripts/dl/profile_decode_step.py` | 逐 step host/GPU 时序分解 |
| `scripts/dl/compile_check.py` | torch.compile 融合正确性验证 |
| `scripts/dl/vllm_features_only.py` | vLLM 服务 A/B 对比 |
| `scripts/dl/compare_results.py` | 按 commit 追踪基准结果 |
| `run_sglang.sh` compare 子命令 | 一键 SGLang vs. vLLM 对比 |

`compare_results.py`（366 行）按 commit 追踪基准结果并按 vLLM runner 版本（MRV1 / MRV2）分组，支持跨开发历史的回归追踪。

---

## 7. 文档

本周产出的技术文档：

- `docs/dl/sglang-dlin-decode-gap-debug-blog.md` / `.zh.md` — Decode gap 根因分析（中英文）
- `docs/dl/sglang-dlin-decode-gap-profiling-blog.md` — 含 host 端时序分解的详细 profiling
- `docs/dl/sglang-vs-vllm-showcase-dlin.md` — 数据驱动的 SGLang vs. vLLM 对比
- `docs/dl/sglang-vs-vllm-features-json-prefix-dlin.md` — JSON prefix 缓存分析
- `docs/dl/torch-compile-phase2-debug-blog.md` — torch.compile Phase II 调试日记
- `docs/dl/torch-compile-phase2-design.md` — Phase II 架构设计
- `docs/dl/torch-compile-stage0-1-plan.md` — Stage 0 和 Stage 1 融合计划
- `docs/dl/dflash-on-dlin-debug-blog.md` — DFlash 调试实录
- `docs/dl/dlin-sglang-mtp-vs-ngram-report.md` — NGRAM vs. DFlash vs. MTP 加速矩阵
- `docs/dl/sglang-port-vllm-kernels-plan.md` — 完整 kernel 移植规划
- `docs/dl/torch-compile-full-support-plan.md` — torch.compile 长期路线图
- `docs/dl/blog-sglang-dlin-qwen35-35b-tp4.md` / `.zh.md` — Qwen3.5-35B 部署指南（中英文）

架构图（`docs/dl/figures/fig1-architecture.svg`、`fig2-latency-decomposition.svg`、`fig3-optimization-journey.svg`）提供中英文版本。

---

## 8. 下一步计划

近期路线图：

- **Kernel 解耦 Phase 5**：将剩余 attention 相关 kernel 从 `vllm._dl_C` 移植，完成依赖移除。
- **投机解码动态调度**：利用已完成的加速矩阵，构建基于 workload 特征在 NGRAM / DFlash / MTP 间动态切换的调度器。
- **DFS（Draft-First Speculation）**：集成 DFS draft 路径以进一步提升 decode 加速。
- **Decode gap 收敛**：评估 in-process tokenizer 方案以消除 6.2ms ZMQ IPC 开销（适用于进程隔离需求较弱的部署场景）。
- **torch.compile Phase II Stage 2–3**：将融合覆盖扩展至 MLP 和 attention 子图，目标是全图编译。
- **CI 集成**：利用 `compare_results.py` 追踪基础设施实现自动化的基准回归检测。

---

## 致谢

本阶段工作主要由谢少波完成。感谢DLIN GPU 工具链团队在 DFlash 和 MTP 调试过程中提供的技术支持，感谢 SGLang 团队在代码审查和架构讨论中的贡献。

所有代码位于 `dl-main` 分支。欢迎社区贡献和反馈。
