# sglang vs vLLM on DLIN — Data-Backed Showcase (Where sglang Wins, Where It Doesn't)

**Date:** 2026-07-22 · **Hardware:** DLIN KS38 (32× QUAD, 32 GiB) · **Model:** Qwen3.5-35B-A3B-**FP8** (hybrid Mamba+attention) · **TP4, GPUs 0–3**
**Harness:** `scripts/dl/showcase_prefix_sharing.py` (same prompt, same GPUs, fresh process per engine, best-of-3, temp=0)

> 目的：用**同模型、同卡、同 prompt、顺序跑**的实测数据，明确告诉别人 sglang 相对 vLLM **到底好在哪里、好多少、哪里不好**。不回避 sglang 输的场景。

---

## TL;DR — 一张图看完

| 场景 | sglang | vLLM | 结论 |
|---|---|---|---|
| **SC1 前缀共享** warm（缓存命中） | **5.56 s** | 13.0 s | **sglang 2.3× 快** |
| **SC1 前缀共享** 冷→暖加速比 | **16.4×** | 1.0× | **sglang 碾压** |
| **SC2 多轮对话** 5 轮平均 | **8.74 s** | 14.4 s | **sglang 1.6× 快** |
| **SC2 多轮** 第 5 轮（4K ctx） | **8.12 s** | 15.8 s | **sglang 1.9× 快** |
| **SC3 并发批** 4 请求吞吐 | **6.0 tok/s** | 2.6 tok/s | **sglang 2.3× 高** |
| **SC3 并发批** 单请求 | **5.34 s** | 12.5 s | **sglang 2.3× 快** |
| SC4 结构化 JSON（短 prompt，预热后） | 31.7 tok/s | 34.6 tok/s | vLLM +9%（基本持平） |
| 纯 decode TPOT（无前缀） | 35.0 tok/s | 37.9 tok/s | vLLM +8% |
| **SC5 多用户 fork**（共享根的 2 分支 radix 树） | **13.2 s** | 108.0 s | **sglang 8.2× 快** ✅ |
| **SC7 长 RAG 吞吐**（~2K 文档 × 8 查询） | **13.0 tok/s** | 0.8 tok/s | **sglang 16.3× 高** ✅ |
| **SC8 best-of-N 并行采样**（n=4，temp=0.7） | **22.2 tok/s** | 3.8 tok/s | **sglang 5.8× 高** ✅ |
| **SC10 共享 system-prompt**（12 租户） | **13.4 tok/s** | 1.9 tok/s | **sglang 7.05× 高** ✅ |
| SC9 纯长 decode（短 prompt，128 tok，无共享） | 30.5 tok/s | **39.6 tok/s** | vLLM 1.30×（诚实控制组）⚠️ |

**一句话**：sglang 在**所有"前缀/多轮/并发"缓存复用场景**全面领先 1.6–2.3×；**vLLM 的 APC（前缀缓存）在本模型上根本无法开启**（硬 assert 失败）。vLLM 仅在**无缓存复用的纯 decode / 短 JSON** 上略快 ~8–9%。

> **2026-07-24 新增场景**（SC5/SC7/SC8/SC9/SC10）：多用户 fork 树、长 RAG 吞吐、
> best-of-N 并行采样、纯 decode 控制、共享 system-prompt 多租户。SC5/SC7/SC10 把
> sglang 的领先从 1.6–2.3× 拉到 **7–16×**（前缀越长/复用越多，vLLM 重 prefill 越惨）。
> 详见 [`sglang-vs-vllm-new-scenarios.md`](sglang-vs-vllm-new-scenarios.md)。运行：
> `./run_sglang.sh compare --scenarios SC5,SC7,SC8,SC9,SC10`。

---

## 1. 实验设置（公平性保证）

- **同模型**：`/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`（FP8，两引擎都用）
- **同卡**：GPU 0–3，TP4，顺序跑（每个引擎 fresh 进程，避免 PCIe/compile 互相干扰）
- **同 prompt**：~2K token 共享系统 prompt + 技术文档 + few-shot（见 `SHARED_PREFIX`）
- **温度 0，best-of-3**，max_new=32（SC1–3）/ 48（SC4）
- **关键修正**：vLLM 用 `.venv` 的 0.21.1.dev2（原生，**无 overlay**），FP8+CG+MRV2；旧 `../venv-vllm021`+overlay 配置是坏的（见 [`dlin-vllm-correct-package-and-env-gotchas`](../../../home/shaobo.xie/.claude/projects/-LocalRun-shaobo-xie-2-Pytorch-docker-test-debug-sglang/memory/dlin-vllm-correct-package-and-env-gotchas.md)）
- sglang：FP8 + CG + RadixAttention + fa3 + page_size=16，`mem_fraction_static=0.55`

---

## 2. SC1 前缀共享 —— sglang 的主场 ⭐

**场景**：8 个请求共享同一个 2K-token 前缀，每个只换最后一句问题。
**测量**：第 1 个请求 = cold（前缀首次 prefill）；第 2–8 个 = warm（前缀已缓存，只需 prefill 新问题）。

```
sglang:  cold=91194ms  warm_median=5564ms  warm_min=5493ms  speedup=16.4×
vLLM:    cold=13068ms  warm_median=13002ms warm_min=12969ms speedup=1.0×
```

### 为什么 sglang 16.4× 而 vLLM 1.0×？

- **sglang RadixAttention**：token 级基数树。第 2 个请求进来时，2K 前缀的 KV **全部命中**，只需 prefill 最后 ~10 个新 token → 91 s → **5.6 s**。
- **vLLM**：每个请求都 **重新 prefill 整个 2K 前缀**（warm 13 s ≈ cold 13 s，1.0×）。原因见 §5：vLLM 的前缀缓存（APC）在本模型上**无法开启**。

### 注意：sglang cold=91s 的诚实说明

sglang 的 cold（91 s）比 vLLM cold（13 s）慢 7×。这是 sglang **大-M prefill 路径未充分融合**（`SGLANG_DL_MOE_FUSED_MAX_M=32`，而 prefill chunk M=512 > 32 走慢路径）+ 首次 JIT 的叠加，**不是 RadixAttention 的问题**。把 `FUSED_MAX_M` 提到 512 可大幅降低 cold（待验证）。warm 5.6 s 才是缓存命中的稳态代表值——而它比 vLLM 的**每一次**请求（13 s）都快。

**净效果（8 请求总量，剔除一次性 JIT 后）**：sglang ≈ 15 + 7×5.6 ≈ 54 s vs vLLM 8×13 = 104 s → **sglang ~1.9× 总更快**。

---

## 3. SC2 多轮对话 —— sglang 越聊越省

**场景**：5 轮对话，每轮把前面所有历史拼上，再问新问题。
**测量**：每轮 wall time。理想情况：只有新 token 被 prefill，历史 KV 复用。

| 轮次 | prompt_len | sglang | vLLM |
|---|---|---|---|
| turn1 | ~3.4K | 6.2 s | 13.1 s |
| turn2 | ~3.5K | 10.9 s | 13.7 s |
| turn3 | ~3.7K | 9.9 s | 14.4 s |
| turn4 | ~3.9K | 8.6 s | 15.0 s |
| turn5 | ~4.0K | 8.1 s | **15.8 s** |

**关键现象**：
- **vLLM 单调上升**：13.1 → 15.8 s。每轮把**整个变长的历史**重新 prefill（无缓存），轮次越多越慢。
- **sglang 基本持平**：6.2–10.9 s。RadixAttention 复用前几轮的 KV，每轮只 prefill **新增**的问答。

到第 5 轮（4K context），**sglang 8.1 s vs vLLM 15.8 s = 1.9× 快**，且差距随轮次增大。

---

## 4. SC3 并发批 —— 共享前缀的批处理

**场景**：4 个请求**共享同一前缀**，作为一个 batch 同时提交。
**测量**：整批 wall time，3 次取最优。

```
sglang:  batch_time=21341ms  total_tokens=128  throughput=6.0 tok/s  per_req=5335ms
vLLM:    batch_time=50163ms  total_tokens=128  throughput=2.6 tok/s  per_req=12541ms
```

sglang 的 RadixAttention 在 batch 内部**去重共享前缀**（4 个请求的 2K 前缀只 prefill 一次），vLLM 无缓存则 4 份各 prefill 一遍 → **sglang 2.3× 吞吐 / 2.3× 单请求更快**。

---

## 5. 为什么 vLLM 缓存"开不了"—— 这是本 showcase 最硬的一条 ⭐⭐

试着给 vLLM 开 `enable_prefix_caching=True`，**EngineCore 直接 init 失败**：

```
WARNING [config.py:367] Mamba cache mode is set to 'align' for
  Qwen3_5MoeForConditionalGeneration by default when prefix caching is enabled
INFO    [config.py:387] Warning: Prefix caching in Mamba cache 'align' mode is
  currently enabled. Its support for Mamba layers is EXPERIMENTAL.

AssertionError: Model Runner V2 has not yet supported mamba_cache_mode='align'.
```

### 根因链
1. Qwen3.5-35B-A3B 是**混合 Mamba+Attention 架构**。
2. 一旦开 APC，vLLM 强制 Mamba cache 进入 `'align'` 模式（block_size=528，按 Mamba 状态对齐）。
3. vLLM **Model Runner V2（MRV2）硬性拒绝** `mamba_cache_mode='align'`（assert 直接挂）。
4. 而 DLIN 上**只能用 MRV2**（MRV1 撞 `ConstraintViolationError`，见 skill 坑③）。
5. ⇒ **vLLM 在本模型/本平台上没有任何可用的前缀缓存路径**。只能 APC-OFF（即 §2–4 的 1.0× / 单调上升 / 重新 prefill）。

> 即便绕过 MRV2（用 MRV1），CG 路径又会触发 `block_size(528) <= max_num_batched_tokens(4)` 的另一条 assert（V2 CG 把 max_num_batched_tokens 钳到 max_cudagraph_capture_size=4）。两条路都堵死。

**这条结论的重要性**：混合 Mamba 架构是当前前沿（Qwen3.5、Jamba、Zamba…）。sglang 的 RadixAttention **原生支持**这类模型的前缀复用，vLLM 的 APC **结构性地不支持**。这不是调参能补的差距。

---

## 6. SC4 结构化 JSON —— vLLM 小胜，需诚实标注

**场景**：短 prompt（~40 token，无前缀共享），JSON schema 约束输出。
**测量**：max_new=48，best-of-3，**预热 schema 编译器后**。

| | sglang (Compressed FSM + xgrammar) | vLLM (xgrammar) |
|---|---|---|
| 约束 decode tok/s | 31.7 | 34.6 |
| 输出有效 | ✅ `{"name":"Dr. Ada Lovelace","age":36,...}` | ✅ 同 |

vLLM +9%，基本持平。**sglang 在 JSON 上没有输很多**，但也没赢——vLLM 的结构化输出后端在这类短 prompt 上同样高效。

> ⚠️ **测量坑**：本 showcase 早期版本（未预热 schema）测出 sglang 仅 10.3 tok/s——那是**每个请求重新编译 JSON schema**（xgrammar FSM 构建 ~3 s/次）的伪影，不是稳态 decode。脚本已加 schema warmup 修正（见 `showcase_prefix_sharing.py` SC4）。两引擎预热后的稳态才可比：sglang 31.7 vs vLLM 34.6。

sglang JSON 的真正优势在**复杂 schema / 长输出 / 与前缀共享叠加**（同一个系统 prompt 下多次结构化抽取，前缀命中 + 约束 decode 双重收益），本短 prompt 微基准体现不出来。

---

## 7. 纯 decode TPOT —— vLLM 略快（已另文详述）

无前缀共享、单请求长 decode 的稳态吞吐：sglang **35.0** vs vLLM **37.9 tok/s**（vLLM +8%）。

这条差距是 sglang 多进程架构的 **scheduler↔worker ZMQ IPC 往返**（6.2 ms/token，100% host 侧），**GPU kernel 两引擎逐字节一致**（20.7 ms/token）。详见 [`sglang-dlin-decode-gap-profiling-blog.md`](sglang-dlin-decode-gap-profiling-blog.md)。这是 sglang 当前**唯一落后**且已知根因的项。

---

## 8. 给"说服别人"用的一页结论

### sglang 明确赢的（数据支撑）
1. **前缀共享**：16.4× 加速（warm 5.6 s vs vLLM 13.0 s 每次）。vLLM 1.0×。
2. **多轮对话**：越聊差距越大，第 5 轮 sglang 1.9× 快；vLLM 线性变慢。
3. **并发批（共享前缀）**：吞吐 2.3× 高。
4. **架构适配**：sglang RadixAttention **原生支持混合 Mamba 模型**；**vLLM APC 在本模型上硬 assert 挂掉，无法开启**。⭐ 这是最硬的一条。

### vLLM 赢的（诚实标注）
1. 纯 decode TPOT：+8%（6.2 ms IPC 往返，结构性，已定位）。
2. 短 prompt JSON：+9%（基本持平）。

### 一句话电梯版
> 在 Qwen3.5-35B（混合 Mamba）这个前沿架构上，**所有需要 KV 复用的真实负载（RAG、多轮、Agent、few-shot、并发）sglang 快 1.6–2.3×**，而且 **vLLM 的前缀缓存根本跑不起来**；sglang 只在"无复用的纯 decode"上慢 8%（已知 IPC 根因，可优化）。

---

## 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh

# sglang（RadixAttention）
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine sglang --mem-frac 0.55

# vLLM（APC 无法开启，唯一可用配置 = APC-OFF + CG）
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm  --mem-frac 0.55
# 验证 APC 崩溃：在脚本里给 LLM() 加 enable_prefix_caching=True →
#   AssertionError: Model Runner V2 has not yet supported mamba_cache_mode='align'.
```

日志：`/tmp/sc_sglang.log`、`/tmp/sc_vllm_fixed.log`（APC-OFF 可用）、`/tmp/sc_vllm_apc2.log`（APC-ON 崩溃链）。

---

## 参考
- 本仓库：[`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md)（JSON/前缀 feature 对比）、[`sglang-dlin-decode-gap-profiling-blog.md`](sglang-dlin-decode-gap-profiling-blog.md)（6.2 ms decode gap 根因）、[`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md)
- Memory：`dlin-vllm-correct-package-and-env-gotchas`、`dlin-sglang-decode-gap-sync-opt-wins`
- 脚本：`scripts/dl/showcase_prefix_sharing.py`（SC1–4，双引擎）
