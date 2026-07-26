# SGLang on DLIN: 投机解码、Kernel 独立与推理优化

SGLang Team · 2026 年 7 月 22 日

过去一周，我们在 DLIN（DLIN GPU）推理引擎上完成了三项系统级改进：**投机解码通路全线打通**、**Kernel 从 vLLM 完全独立**、以及 **torch.compile 融合编译第二阶段落地**。以下逐一展开，每项工作附带代码栈解析和设计意图说明。

---

**核心指标摘要**

| 指标 | 数值 |
|------|------|
| DFlash 代码场景加速比 | 2.99× |
| Prefix Sharing cold→warm 加速比 | 16.4× |
| SGLang vs vLLM 前缀复用场景领先 | 1.6–2.3× |
| 自研 flash-attn decode 吞吐 | 35.0 tok/s |
| 与 vLLM 的 decode gap（已定位） | 6.2 ms/token，100% ZMQ IPC |
| Kernel 解耦 phase 完成数 | 4/5 |
| 新增代码 | 79 文件，+9,462 / −268 行 |

---

## 1. 投机解码：DFlash 与 MTP

### 1.1 问题背景

投机解码（Speculative Decoding）的核心思路是用一个轻量 draft 模型先生成若干候选 token，再用目标模型做批量验证。如果 draft 的接受率高，就能用一次验证的开销换取多个 token 的产出，从而加速 decode。这个技术在 DLIN 上的挑战在于：DLIN 的算子实现和 CUDA graph 机制与 NVIDIA 不同，需要针对 DLIN 的 runtime 做适配。

### 1.2 DFlash：正确性验证与 dl_chunk 优化

**代码栈**

```
scripts/dl/dflash_correctness.py      # 正确性验证 harness
python/sglang/srt/speculative/dflash_worker_v2.py   # DFlash worker 实现
scripts/dl/dl_chunk_test.py           # dl_chunk prefill 测试
```

**做了什么**

首先，我们为 DFlash 构建了完整的正确性验证流水线。`dflash_correctness.py` 将 DFlash 的输出 logprob 分布与非投机 baseline 逐 token 做统计比对，确认 DLIN 上的 DFlash 实现精度正确——这一步不可跳过，因为投机解码一旦出现精度偏差，会在长文本生成中不断累积。

其次，我们发现了 DFlash draft 路径的瓶颈：draft 模型需要为每个投机位置分别做一次前向计算，这部分开销随 spec 步数线性增长。我们的解法是 `dl_chunk`——将多个 draft 位置拼接为一个 batch 做一次前向。核心改动在 `dl_chunk_test.py` 中验证通过，prefill 吞吐提升约 +5%。

```
大白话：原来 draft 要一步一步推，现在把好几步打包成一批一起算，省掉了反复启动 kernel 的开销。
```

### 1.3 MTP：FROZEN_KV_MTP 修复与端到端跑通

**代码栈**

```
python/sglang/srt/models/qwen3_5_mtp.py                    # MTP 模型定义
python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py  # CUDA graph runner
```

**做了什么**

MTP（Multi-Token Prediction）是 Qwen3.5 的原生多头预测机制，draft 头与目标模型共享主干。在 DLIN 上，我们用 CUDA graph 来加速 MTP 的 decode 循环——CUDA graph 可以一次性 capture 整个 kernel launch sequence，免除每次迭代的 CPU launch 开销。

但 `FROZEN_KV_MTP` 路径存在三个问题：graph capture 时 draft head 的 KV cache tensor 地址不稳定、跨 replay 迭代的指针漂移、以及变长 draft 输出在固定 graph 形状下的边界处理。这些修复让 MTP 在 DLIN 上第一次跑通了端到端。

```
大白话：CUDA graph 就像把一连串 GPU 指令录下来，以后直接"回放"而不用一句一句重新下发。
但录的时候 tensor 地址不能变，变了回放就会读到错误的数据。
我们修的就是这些地址漂移的问题。
```

### 1.4 加速矩阵

| 模式 | 代码场景 | 通用场景 |
|------|---------|---------|
| NGRAM | 1.5× | 1.2× |
| DFlash | 2.99× | 1.8× |
| MTP | 2.1× | 1.6× |

不同场景有各自的优势模式，这也正是后续动态调度器的切入点。

---

## 2. Kernel 独立：Phase 1–4

### 2.1 问题背景

SGLang on DLIN 的 kernel 长期寄生在 vLLM 的 `_custom_ops` 扩展（即 `_dl_C.so`）中。这意味着每次 vLLM 发布新版本，DLIN 的 kernel 都要跟着适配编译；反之，DLIN 的 kernel 改动也无法独立演进。这本质上是一个**编译时依赖的耦合问题**。

### 2.2 解耦方案

**代码栈**

```
sgl-kernel/csrc/elementwise/gemma_rmsnorm_dl.cu    # gemma_rms_norm CUDA 实现
sgl-kernel/csrc/common_extension_dl.cc              # DL 扩展注册入口
sgl-kernel/setup_dl.py                              # DL 构建配置
python/sglang/srt/layers/rotary_embedding/base.py   # rotary 切换
python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py  # MoE 切换
python/sglang/srt/layers/quantization/marlin_utils.py              # marlin 切换
python/sglang/srt/layers/layernorm.py                              # layernorm 切换
```

**做了什么**

解耦分四个 phase，按 kernel 的依赖深度逐一清理：

<center>

**Phase 1 — rotary + fused MoE**

| 之前 | 之后 |
|------|------|
| `torch.ops._dl_C.rotary_embedding(...)` | `sglang/srt/layers/rotary_embedding/base.py` 本地实现 |
| `torch.ops._dl_C.fused_moe(...)` | `triton_utils/fused_moe.py` 本地 TMA 分发 |

**Phase 2 — marlin GEMM**

| 之前 | 之后 |
|------|------|
| `vllm.model_executor.layers.quantization.marlin` | `marlin_utils.py` 本地 shim 封装 |

**Phase 3 — suppress_other_loggers**

| 之前 | 之后 |
|------|------|
| `vllm.logger` | sglang 本地 logger 实现 |

**Phase 4a/4e — gemma_rms_norm**

| 之前 | 之后 |
|------|------|
| `vllm._dl_C.gemma_rms_norm` | `sgl_kernel.gemma_rms_norm`（CUDA 223行，向量化内存访问） |

</center>

```
大白话：以前每次跑 sglang 都要先装 vLLM，因为 kernel 都在 vLLM 的动态库里。
现在把 kernel 一个个搬到自己家（sgl-kernel），以后 vLLM 升级跟我们没关系了。
```

---

## 3. torch.compile Phase II：PostGradPassManager 与算子融合

### 3.1 问题背景

在 Phase I 中，我们把 DLIN 的大算子（fused MoE、flash attention）保留为 opaque 操作，避免 torch.compile 将它们分解成大量细粒度小 op 导致编译爆炸。Phase II 的目标是在这个限制下，对小算子序列（activation → quantize → matmul）做融合，减少 GPU kernel launch 次数和中间张量的显存读写。

### 3.2 融合框架

**代码栈**

```
python/sglang/srt/compilation/passes/fusion/__init__.py        # fusion pass 注册
python/sglang/srt/compilation/passes/fusion/dl_act_quant_fusion.py   # act-quant 融合
python/sglang/srt/compilation/passes/fusion/dl_matcher_utils.py      # 匹配工具
python/sglang/srt/compilation/passes/fusion/dl_pattern_matcher.py    # 模式匹配引擎
python/sglang/srt/compilation/pass_manager.py                  # PostGradPassManager
python/sglang/srt/compilation/torch_compile_decoration.py      # 双策略调度
tests/dl/test_compile_fusion.py                                # 融合正确性测试
scripts/dl/compile_check.py                                    # 编译检查
```

**做了什么**

我们在 `PostGradPassManager` 中插入了一个 fusion pipeline，包含三个 pass：

1. **`ActivationQuantFusionPass`**：匹配 `silu_and_mul → quant` 子图，当 activation 的输出直接接量化操作时，将两者合并为一个 fused kernel。减少了一次 global memory 读写。

2. **`RMSNormQuantFusionPass`**：匹配 `rms_norm → quant` 子图，原理同上。

3. **`PatternMatcherPass`**：通用模式匹配器，支持声明式规则定义，方便后续添加新的融合模式。

每个 pass 的实现分为三步：遍历 FX graph 找到匹配子图 → 验证正确性（tensor dtype、shape 兼容性）→ 替换为融合 op。单元测试覆盖了每种融合模式的 eager vs compile 一致性。

```
大白话：torch.compile 能把一连串小算子"撮合"成一个，省掉中间结果的读写。
但 DLIN 的大算子已经是黑盒了，所以只能在小算子之间做融合。
我们做了 activation → quantize 的撮合，类似把"算完再量化"变成"边算边量化"。
```

### 3.3 双策略 Compile + CUDA Graph

| Batch 大小 | 策略 | 原因 |
|-----------|------|------|
| 1–32 | torch.compile | 融合收益大于编译开销 |
| 32+ | CUDA graph | graph capture 一次，零 CPU launch 开销 |

`torch_compile_decoration.py` 根据当前 batch size 自动选择执行路径。设 `--enable-torch-compile` 即可激活。

### 3.4 设计意图

Phase II 的架构选择体现了一个权衡：DLIN 的 fused 大算子（MoE、attention）不能 decompose——但小算子（norm、activation、quant）的融合仍有优化空间。后者的优化不需要改动 CUDA kernel，完全在 FX graph 层面完成，对系统稳定性影响小、迭代速度快。

后续 Stage 2–3 的目标是将融合范围扩大到跨算子边界（如 attention 输出 → MoE 输入之间的过渡），这需要引入新的 DLIN 专用 fusion kernel。

---

## 4. 自研 Flash Attention 与 Decode 时延分解

### 4.1 问题背景

DLIN 的 flash attention 路径此前依赖 `vllm_flash_attn` 包（PyPI），该包将 CUDA kernel 注册在 `_vllm_fa2_C` namespace 下。当 sglang 和 vLLM 共存于同一个 `.venv` 时（`SGLANG_DL_MOE_VLLM=1` 场景），两个 `.so` 文件同时注册同一个 `_vllm_fa2_C` 符号 → `c10::Error` SIGABRT。这是一个典型的 **TORCH_LIBRARY namespace 冲突**问题。

### 4.2 解决方案

**代码栈**

```
python/sglang/srt/layers/attention/sgl_flash_attn.py      # sglang 自封装
python/sglang/srt/layers/attention/dl_flash_attn.py        # attention backend
python/sglang/srt/layers/attention/flashattention_backend.py
sgl-kernel/csrc/                                          # flash-attn 编译
```

**做了什么**

编译 sglang 自有的 `flash_attn_2_cuda.so`，namespace 改为 `_sgl_fa2_C`，并提供 Python wrapper。attention backend 改为从 sglang wrapper 导入：

```
之前: dl_flash_attn.py → import vllm_flash_attn → torch.ops._vllm_fa2_C
之后: dl_flash_attn.py → import sgl_flash_attn → torch.ops._sgl_fa2_C
```

修复过程中遇到了 4 层嵌套问题：

| 层级 | 问题 | 修复 |
|------|------|------|
| L1 | `.so` namespace 冲突 | 重编译为 `_sgl_fa2_C` |
| L2 | `easy-install.pth` 导致加载旧 `.so` | 注释 dev install 路径 |
| L3 | `.venv` 中有两份 `_vllm_fa2_C.so` | 隐藏 vLLM 内置副本 |
| L4 | `fp8.py` 硬编码 sys.path 注入 | 删除注入，统一从 `.venv` 导入 |

```
大白话：两个动态库都叫同一个名字，Python 加载第二个的时候就会崩溃。
解决办法是把 sglang 用的那份改个名字，从此井水不犯河水。
```

### 4.3 Decode 时延分解

**代码栈**

```
scripts/dl/profile_decode_step.py   # decode step 时序分析
docs/dl/sglang-dlin-decode-gap-profiling-blog.md  # 详细报告
```

**做了什么**

逐一测量 decode step 的每个子阶段耗时，得到以下分解：

| 子阶段 | 耗时 | 累计 | 说明 |
|--------|------|------|------|
| GPU kernel（attention + MoE + MLP） | 20.7 ms | 20.7 ms | 与 vLLM 完全一致 |
| Host 调度（scheduler） | ~3 ms | ~23.7 ms | batch 组装 + radix 操作 |
| ZMQ IPC（scheduler → tokenizer） | ~6.2 ms | ~29.9 ms | 跨进程序列化+传输 |
| **总计** | **~29.9 ms** | | → 对应 ~33.5 tok/s |

核心发现：GPU kernel 部分与 vLLM **完全相同**（20.7 ms），差距 100% 来自 ZMQ IPC。这是因为 sglang 采用分离式架构（scheduler 与 tokenizer 分属不同进程），而 vLLM 采用进程内架构。这是架构取舍的代价，而非可优化的 bug。

```
大白话：GPU 干活的速度其实和 vLLM 一样快。差距在 CPU 这边——
sglang 的调度器和分词器不在一个进程里，每次通信要走 ZMQ 发一轮消息。
这不是 bug，是分离式架构的代价——换来了更强的隔离性和可扩展性。
```

---

## 5. SGLang vs. vLLM：同硬件、同模型、公平对比

### 5.1 实验设计

| 维度 | 方案 |
|------|------|
| 硬件 | DLIN KS38, GPU 0–3, TP4 |
| 模型 | Qwen3.5-35B-A3B-FP8（同一份权重） |
| 顺序 | 顺序执行，每个引擎 fresh 进程 |
| 重复 | best-of-3，温度 0 |
| 工具 | `scripts/dl/showcase_prefix_sharing.py` + `run_sglang.sh compare` |

### 5.2 数据

| 场景 | SGLang | vLLM | 倍数 |
|------|--------|------|------|
| SC1 前缀共享 warm（2K prefix, cache hit） | **5.56 s** | 13.0 s | **2.3×** |
| SC1 cold→warm 加速比 | **16.4×** | 1.0× | **16.4×** |
| SC2 多轮对话 5 轮平均 | **8.74 s** | 14.4 s | **1.6×** |
| SC2 第 5 轮（4K ctx） | **8.12 s** | 15.8 s | **1.9×** |
| SC3 并发批 4 请求吞吐 | **6.0 tok/s** | 2.6 tok/s | **2.3×** |
| SC4 结构化 JSON（短 prompt） | 31.7 tok/s | 34.6 tok/s | vLLM +9% |
| 纯 decode TPOT（无前缀） | 35.0 tok/s | 37.9 tok/s | vLLM +8% |

```
大白话：有缓存复用场景——SGLang 碾压，1.6 到 16 倍的领先。
无缓存复用的单请求场景——vLLM 略快 8-9%。
因为真实世界的请求大多有重复前缀（系统 prompt、few-shot、多轮对话），SGLang 更占优势。
```

### 5.3 为什么 vLLM 的 APC 没起作用

vLLM 有 Automatic Prefix Caching（APC）功能，理论上也能在前缀共享场景中加速。但在 Qwen3.5-35B-A3B-FP8 这个模型上，APC 无法开启——vLLM 内部的 assert 在 hybrid Mamba+Attention 架构下触发失败。这是模型兼容性问题，不是原理上的限制。

```
大白话：vLLM 也有前缀缓存，但我们这个模型比较新（Hybrid Mamba+Attention），
vLLM 的缓存功能在这个架构上还没适配好，硬 assert 就挂掉了。
```

---

## 6. 基准测试基础设施

**代码栈**

| 脚本 | 功能 |
|------|------|
| `run_sglang.sh compare` | 一键 SGLang vs. vLLM 对比 |
| `scripts/dl/showcase_prefix_sharing.py` | 前缀共享多场景 benchmark |
| `scripts/dl/dflash_correctness.py` | DFlash 正确性验证 |
| `scripts/dl/mtp_correctness.py` | MTP 正确性验证 |
| `scripts/dl/dl_chunk_test.py` | Chunked prefill 延迟 |
| `scripts/dl/prefill_latency_test.py` | Prefill 延迟基准 |
| `scripts/dl/profile_decode_step.py` | Decode step 时序分解 |
| `scripts/dl/compile_check.py` | torch.compile 融合验证 |
| `scripts/dl/compare_results.py` | 按 commit 追踪基准 |
| `scripts/dl/vllm_features_only.py` | vLLM 独立服务 |

```
大白话：以前比性能和找回归要靠手动跑、手动记。
现在一个脚本跑完，结果按 commit 自动归档，谁改了什么、性能涨了跌了一目了然。
```

---

## 7. 文档

本周产出的技术文档按主题分组：

**性能分析**
- `docs/dl/sglang-dlin-decode-gap-debug-blog.md` — Decode gap 根因分析（中英文）
- `docs/dl/sglang-dlin-decode-gap-profiling-blog.md` — Host 端时序分解 profiling

**对比报告**
- `docs/dl/sglang-vs-vllm-showcase-dlin.md` — SGLang vs. vLLM 数据对比
- `docs/dl/sglang-vs-vllm-features-json-prefix-dlin.md` — JSON prefix 缓存分析
- `docs/dl/dlin-sglang-mtp-vs-ngram-report.md` — NGRAM/DFlash/MTP 加速矩阵

**系统设计**
- `docs/dl/torch-compile-phase2-debug-blog.md` — Phase II 调试日记
- `docs/dl/torch-compile-phase2-design.md` — Phase II 架构设计
- `docs/dl/torch-compile-stage0-1-plan.md` — Stage 0/1 融合计划
- `docs/dl/torch-compile-full-support-plan.md` — 长期路线图
- `docs/dl/sglang-port-vllm-kernels-plan.md` — Kernel 移植规划

**经验总结**
- `docs/dl/dflash-on-dlin-debug-blog.md` — DFlash 调试实录
- `docs/dl/blog-sglang-dlin-qwen35-35b-tp4.md` — Qwen3.5-35B 部署指南（中英文）

架构图 `docs/dl/figures/fig{1,2,3}-*.svg` 提供中英文版本。

---

## 8. 下一步计划

**短期（1–2 周）**

| 项目 | 目标 | 优先级 |
|------|------|--------|
| Kernel 解耦 Phase 5 | attention 相关 kernel 从 `_dl_C` 移出 | P0 |
| Decode gap closure | 评估 in-process tokenizer 消除 6.2ms IPC | P1 |
| CI 集成 | 自动化基准回归检测 | P1 |

**中期（3–6 周）**

| 项目 | 目标 |
|------|------|
| 投机解码动态调度 | 基于 workload 特征在 NGRAM/DFlash/MTP 间自动切换 |
| DFS 集成 | 引入 Draft-First Speculation 路径 |
| torch.compile Stage 2–3 | 融合覆盖扩展到 MoE → MLP 过渡 |

---

## 致谢

本阶段工作主要由谢少波完成。感谢DLIN GPU 工具链团队在 DFlash 和 MTP 调试中的技术支持，感谢 SGLang 团队的代码审查与架构讨论。

所有代码位于 `dl-main` 分支。
