# sglang vs vLLM on DLIN — 新场景调研（SC5/SC7/SC8/SC9/SC10）

> 配套文档：[`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md)
>（SC1–SC4）。本文记录了 **2026-07-24 调研**——用新的工作负载模式扩展展示，
> 所用方法学，以及每个新场景的诚实胜负记录。
> **2026-07-27 更新：** MRV1 APC+CG-on 已修复，SC1–SC3 三引擎数据见
> [`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md)。以下场景（SC5+）尚未用 MRV1 测试。
>
> **严谨性审计：** 每个场景的公平性在
> [`sglang-vs-vllm-rigor-analysis.md`](sglang-vs-vllm-rigor-analysis.md) 中经过压力测试——
> 包括 SC6 原始 prefill 对比探针（sglang 49 vs vLLM 76 tok/s ⇒ 胜出 100% 来自
> RadixAttention 缓存，而非更快的原始 prefill）和 SC8b cold-best-of-N 探针
>（SC8 的 5.84× = 缓存 3.9× × 单调用 2.0×）。

## TL;DR

_完成（运行 r008，commit ba1bab09de）。4 个 sglang 胜出（5.8–16.3×）+ 1 个诚实的 vLLM 解码胜出（SC9，控制组）。_

| 场景 | 工作负载（共享结构形状） | sglang | vLLM-MRV2 | 结论 |
|----------|-----------------------------------|--------|-----------|---------|
| **SC5** 多用户分叉 | 2 用户 × 4 轮，共享根（radix **树**） | **13.2 s**（1653 ms/轮） | **108.6 s**（13578 ms/轮） | **sglang 快 8.22×** ✅ |
| **SC7** 长 RAG 吞吐 | ~2K token 文档 × 8 顺序查询 | **13.0 tok/s** | **0.8 tok/s**（228 s） | **sglang 高 16.25×** ✅ |
| **SC8** 重复 best-of-N（RLHF 循环） | n=4（temp=0.7），相同提示词复用 | **22.2 tok/s** | **3.8 tok/s**（51 s） | **sglang 5.84×** ⚠️（= 缓存 3.9× × 单调用 2.0×；见 SC8b） |
| **SC9** 纯长解码 | 短提示词 + 128 tok 单流（解码受限） | 30.5 tok/s | **39.6 tok/s** | **vLLM 1.13–1.30×（诚实失利/控制组）** ⚠️ |
| **SC10** 共享系统提示词 | ~0.9K 系统提示词 × 12 租户 | **13.4 tok/s** | **1.9 tok/s**（151 s） | **sglang 高 7.05×** ✅ |

**假设 vs 现实（实测）：**
- SC5 / SC7 / SC8 / SC10 是前缀/KV 复用（或 prefill 密集）的工作负载 →
  sglang **胜出 5.8–16.3×**（vLLM 的 APC 在此混合 Mamba 模型上结构性不支持，
  因此每次重新 prefill；其 prefill 也慢，约 ~75 tok/s）。SC8——预期是 vLLM
  有利的解码受限场景——实际 **被 sglang 以 5.8× 胜出**，因为 vLLM 的 n=4
  prefill+decode 在此较慢。
- SC9（纯解码，无共享结构）是唯一一个 vLLM 诚实胜出的场景：**vLLM 1.30×**
  （39.6 vs 30.5 tok/s）——解码 IPC 差距。没有需要缓存复用的内容时，vLLM
  更快的原始解码胜出。

## 1. 为什么选这些场景（而不是仅仅"更大的 SC1"）

SC1–SC4 已涵盖：独立提示词间的扁平前缀（SC1）、单线多轮对话（SC2）、
单个批处理前缀调用（SC3）、短 JSON（SC4）。新场景各自针对**不同的生产工作负载形状**，
其中 KV 缓存复用（或缺乏复用）起主导作用：

- **SC5 — 多用户分叉（radix 树）。** 两个用户在**共享的系统/上下文根**上各进行
  4 轮对话，在同一引擎中交错执行。这是一个*分支*树（根 → {用户 A 分支，用户 B 分支}）——
  这正是 RadixAttention 命名的结构——不同于 SC1 的扁平前缀或 SC2 的单线。
  生产形状：一个系统提示词背后的多用户/多代理。胜出机制：根 prefill 一次，
  每轮仅扩展增量后缀；vLLM 每轮重新 prefill 整个增长中的对话。
- **SC7 — 长 RAG 吞吐。** 一个 ~2K token 文档 + 8 个不同的顺序查询，
  报告为**聚合解码吞吐量**。比 SC1/SC3 长约 ~2× 的前缀放大了 vLLM
  每查询支付的重新 prefill 成本。生产形状：长共享文档上的 RAG。
- **SC8 — 并行采样（best-of-N）。** 一个提示词 → 单次调用中 n=4 候选补全。
  解码受限控制组：提示词在*两个*引擎中都 prefill 一次后分叉，因此 APC
  不相关，约 ~8% 的解码 IPC 优势决定胜负。生产形状：best-of-N / RLHF 拒绝采样。
- **SC10 — 共享系统提示词吞吐。** 24 个独立的短请求，全部共享一个 ~0.9K
  系统提示词。RadixAttention 投入生产的经典形状（一个聊天机器人角色，多个用户）。
  前缀比 SC7 短约 ~3×，请求多 1.5×。

## 2. 公平性 / 方法学（与 SC1–SC4 相同）

- 相同模型（Qwen3.5/3.6-35B-A3B-FP8）、相同 4 张 GPU、**顺序**执行，
  每个引擎 fresh 进程，温度 0。
- 一次 warmup 轮次后取 **best-of-3** 测量轮次（填充 radix 树）。
  `ignore_eos=True` 用于固定解码长度（干净的每 token 计时）。
- vLLM 以 **APC-关** 运行——MRV2 拒绝 `mamba_cache_mode='align'`。
- 在 DLIN 上 vLLM **MRV2** 是默认 runner。**MRV1 (APC+CG-on) 目前已可运行**
  （2026-07-27 修复 `compilation_config` 冲突后），但不在这些扩展场景中测试。
  见 [`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md) §5 的 MRV1 修复说明。
- 框架：`scripts/dl/showcase_prefix_sharing.py`（扩展版）；驱动
  `./run_sglang.sh compare --scenarios SC5,SC7,SC8[,SC10]`。每个场景在
  非致命守卫内运行，一个场景的崩溃不会波及其他场景。

## 3. 结果

所有数字：Qwen3.5/3.6-35B-A3B-FP8，TP4，相同 4 张 GPU，每个引擎 fresh 进程，
温度 0（SC8 除外 = 0.7 用于真实 best-of-N），一次 warmup 后取 best-of-2，
`ignore_eos=True`。来源：`docs/dl/compare_results.json`（运行 r007 + 全部 5 个重跑）。

| 场景 | sglang | vLLM-MRV2 | sglang 优势 |
|----------|--------|-----------|------------------|
| **SC5** 分叉总计 | 13.2 s（1652 ms/轮） | 108.0 s（13504 ms/轮） | **快 8.2×** ✅ |
| **SC7** 长 RAG 吞吐 | 13.0 tok/s | 0.8 tok/s（227 s） | **高 16.3×** ✅ |
| **SC10** 共享系统提示词 | 13.4 tok/s | 1.9 tok/s（151 s） | **高 7.05×** ✅ |
| **SC8** 并行采样（n=4，temp=0.7） | 22.2 tok/s | 3.8 tok/s（51 s） | **sglang 高 5.8×** ✅ |
| **SC9** 纯长解码（128 tok） | 30.5 tok/s | 39.6 tok/s | **vLLM 1.30×（控制组）** ⚠️ |

**胜出为何如此巨大（sglang vs MRV2）。** MRV2 的 APC 在此混合 Mamba 模型上结构性关闭，
因此它**在每个请求上重新 prefill 共享前缀**。更糟的是，vLLM 在此模型上
测得的 prefill 仅约 ~75 tok/s（远慢于 sglang），因此每次重新 prefill 都很昂贵。
sglang 的 RadixAttention 保留共享 KV，仅扩展微小的每请求后缀。
前缀越长/越共享（SC7 的 ~2K 文档、SC10 的 12 租户、SC5 的增长分支），
差距就越大。

> **2026-07-27 更新：** MRV1 (APC+CG-on) 在 SC1–SC3 上已能跑通且超过 sglang
> 1.7–2.2×。这些扩展场景尚未用 MRV1 测试，但预期 MRV1 也可能显著缩小差距。

**为什么 SC5（8.2×）< SC7（16.3×）。** SC5 每轮的增量后缀更大（生成的助手文本 +
新问题），且只有 8 次生成；SC7 重新 prefill 一个 ~2K 文档 24 次
（warmup + 2×8）。vLLM 更多的重新 prefill 工作量 ⇒ 更大比率。

## 4. 失利记录（供后续优化）

- **SC9 — 纯长解码（唯一诚实的 vLLM 胜出）。** 短（~14 token）提示词 +
  128 token 单流贪心解码。无共享结构 ⇒ 由原始解码 TPOT 主导，
  其中 vLLM 保持着 DLIN IPC 优势：**vLLM 39.6 tok/s vs sglang 30.5 tok/s
  （vLLM 1.30×）**。这是故意的对照组：没有需要缓存复用的内容时，
  vLLM 更快的解码胜出。sglang 的优化路径：缩小解码 IPC 差距
  （重叠调度器 / 减少每步主机同步——参见
  `docs/dl/sglang-dlin-decode-gap-debug-blog.md`）。注意：此处的差距（1.30×）
  比以前"vLLM +8%"更宽，因为展示配置中的 sglang 解码（30.5 tok/s）
  在其 ~35 tok/s 最佳值以下运行。
- **SC8 — 原始贪心 best-of-N 在 vLLM 中无效**（`n must be 1 when using greedy sampling`）。
  通过 `temperature=0.7` 修复（真正的 best-of-N 需要采样）。修复后，
  SC8 是 **sglang 5.8× 胜出**（vLLM 的 n=4 prefill+decode 在此较慢），而非失利。
- MRV1 (APC+CG-on) 修复后，SC1–SC3 上 MRV1 已超过 sglang。这些扩展场景尚未用 MRV1 测试。

## 5. 复现

```bash
# 胜出 + 控制组，两个引擎，相同 GPU（TP4）。约 25-30 分钟（vLLM 重新 prefill）。
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10
# 从缓存重新渲染最后的差距表（无需 GPU 运行）：
./run_sglang.sh compare --show
# 历史（commit 键控的结果日志）：
./run_sglang.sh compare --history
```

## 6. 结论——sglang 胜出的设计空间（在此模型上）

Qwen3.5/3.6-35B-A3B-FP8（DLIN）上可行的 sglang 有利工作负载已由 SC1–SC10
穷尽覆盖。**每个 KV 复用形状都是 sglang 胜出（2.3–16.3×）**，因为 vLLM 的 APC
在此混合 Mamba 模型上结构性关闭（MRV2 拒绝 `mamba_cache_mode='align'`），
所以 vLLM 重新 prefill——且其在此模型上测得的 prefill 很慢（~75 tok/s）。
sglang 的 RadixAttention 保留 KV 并仅扩展后缀。覆盖范围：

- 扁平前缀（SC1）、线性多轮（SC2）、批处理前缀（SC3）——已有，
- 分支多用户分叉树（SC5）、长 RAG 吞吐（SC7）、
- best-of-N 并行采样（SC8）、共享系统提示词多租户（SC10）。

**MRV1 改变了这一画面（2026-07-27）。** SC1–SC3 上 MRV1 (APC+CG-on) 全面超过
sglang 1.7–2.2×——说明**不是 APC 结构性不支持，而是 MRV2 不支持**。MRV1
需要 opt-in 和两个配置坑的处理，但跑通后性能更好。sglang 的独特价值变为：
"零配置就有前缀缓存"，而非"唯一能缓存前缀的引擎"。

当前 sglang 在 DLIN 上的非前缀优化仍受阻：在线并发（FA2 包装器对超过 2 个
重叠 prefill 崩溃）、推测解码（验证质量 bug）、torch.compile（第二阶段未就绪）。
因此下一个*定性新* sglang 胜出需要解除其中一个阻塞（FA2 varlen、spec 质量或
compile 融合）。
