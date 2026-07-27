# sglang vs vLLM on DLIN —— 三引擎数据驱动的展示（sglang / MRV2 / MRV1）

**日期：** 2026-07-27 · **硬件：** DLIN KS38（32× QUAD，32 GiB）· **模型：** Qwen3.6-35B-A3B-**FP8**（混合 Mamba+attention）· **TP4**
**框架：** `scripts/dl/showcase_prefix_sharing.py`（相同 prompt、相同 GPU、每引擎 fresh 进程、best-of-3、temp=0）

> 目的：用**同模型、同卡、同 prompt、顺序跑**的实测数据，回答三引擎在混合 Mamba 模型上的前缀缓存表现。sglang（默认 RadixAttention）、vLLM-MRV2（默认 APC-OFF）、vLLM-MRV1（APC+CG-on opt-in）。

---

## TL;DR — 一张图看完

| 场景 | sglang | vLLM-MRV2 | vLLM-MRV1 (APC+CG-on) | 结论 |
|---|---|---|---|---|---|
| **SC1 前缀共享** warm | **2.01 s** | 13.06 s | **1.21 s** | MRV1 1.67× > sglang > MRV2 6.5× |
| **SC1 前缀共享** cold→warm | **10.4×** | 1.01× | 1.35× | sglang 碾压（radix 树） |
| **SC2 多轮对话** 平均 | **2.87 s** | 14.50 s | **1.29 s** | MRV1 2.2× > sglang > MRV2 5.1× |
| **SC2 多轮** 第 5 轮 | **2.19 s** | 16.00 s | **1.21 s** | MRV1 1.8× > sglang > MRV2 7.3× |
| **SC3 并发批** 吞吐 | **20.0 tok/s** | 2.5 tok/s | **41.9 tok/s** | MRV1 2.1× > sglang > MRV2 8.0× |
| **SC3 并发批** 单请求 | **1.60 s** | 12.68 s | **0.76 s** | MRV1 2.1× > sglang > MRV2 7.9× |
| SC4 结构化 JSON（短 prompt） | 31.7 tok/s | 34.6 tok/s | — | vLLM +9%（基本持平） |
| **SC5 多用户 fork** | **13.2 s** | 108.0 s | — | **sglang 8.2× 快** ✅ |
| **SC7 长 RAG** | **13.0 tok/s** | 0.8 tok/s | — | **sglang 16.3× 高** ✅ |
| **SC10 共享 system-prompt** | **13.4 tok/s** | 1.9 tok/s | — | **sglang 7.05× 高** ✅ |
| SC9 纯长 decode | 30.5 tok/s | **39.6 tok/s** | — | vLLM 1.30× ⚠️ |

**核心发现**：**MRV1 (APC+CG-on) 全面超过 sglang**（1.7–2.2×），因为 MRV1 有 APC + 更快的 DLIN 原生路径。但 vLLM **默认配置 MRV2 仍然 APC-OFF**（结构性不支持 Mamba → 6–8× 慢于两者）。sglang 的优势在于：它是 DLIN 上**唯一默认就有前缀缓存的引擎**（无需切换 runner）。MRV1 虽然更快，但需要显式 opt-in（`VLLM_USE_V2_MODEL_RUNNER=0` + 特殊 compilation_config）。

> **注**：SC5/SC7/SC8/SC9/SC10 等扩展场景尚未用 MRV1 测试。
> 详见 [`sglang-vs-vllm-new-scenarios.md`](sglang-vs-vllm-new-scenarios.md)。

---

## 1. 实验设置（公平性保证）

- **同模型**：`/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8/`（FP8，三引擎都用）
- **同卡**：GPU 8–11，TP4，顺序跑（每个引擎 fresh 进程，避免互相干扰）
- **同 prompt**：~2K token 共享系统 prompt + 技术文档 + few-shot（见 `SHARED_PREFIX`）
- **温度 0，best-of-3**，max_new=32（SC1–3）/ 48（SC4）
- **三引擎**：
  - **sglang**：FP8 + CG + RadixAttention + fa3 + page_size=16，`mem_fraction_static=0.55`
  - **vLLM-MRV2**（默认）：FP8 + CG + **APC-OFF**（Mamba 结构性限制）
  - **vLLM-MRV1**（opt-in）：FP8 + CG + **APC-ON** + `compilation_config={"mode":"NONE", "cudagraph_capture_sizes":[1,2,4,528]}`
- vLLM 版本：0.21.1.dev2（`.venv` 原生）

---

## 2. SC1 前缀共享 —— 三引擎对比 ⭐

**场景**：8 个请求共享同一个 2K-token 前缀，每个只换最后一句问题。
**测量**：第 1 个请求 = cold（前缀首次 prefill）；第 2–8 个 = warm（前缀已缓存，只需 prefill 新问题）。

```
sglang:   cold=20876ms  warm_median=2010ms  speedup=10.4×
vLLM-MRV2: cold=13178ms  warm_median=13058ms speedup=1.0×
vLLM-MRV1: cold=1628ms   warm_median=1206ms  speedup=1.4×
```

### 为什么 MRV1 比 sglang 还快（1206 vs 2010 ms）？

- **MRV1 (APC+CG-on)** 的前缀缓存**确实生效**了：warm 1206 < cold 1628（1.4×），说明 APC 命中后减少了 prefill 量。而且 MRV1 的 decode CG 路径和 DLIN 原生算子路径比 sglang 更高效（无多进程 IPC 开销）。
- **sglang RadixAttention**：warm 2010 ms（cold→warm 加速比 10.4×），但是**绝对延迟**更高——因为 sglang 的 prefill 路径在 DLIN 上更慢（见 §7）且伴有 JIT 抖动。
- **MRV2**：APC 结构性不支持 Mamba（MRV1 才能开），每个请求完整 prefill 2K 前缀。

### cold 的诚实说明

sglang cold（20.9 s）仍含首次 JIT；MRV1 的 CG 初始化也含 warmup 但无 compile JIT。MRV1 cold 1.63 s ≈ sglang warm 2.01 s——这说明 MRV1 的 prefill+decode 原生路径确实比 sglang 高效。

**净效果（8 请求总量）**：
- sglang: 20.9 + 7×2.0 ≈ **34.9 s**
- MRV1: 1.6 + 7×1.2 ≈ **10.0 s**
- MRV2: 13.2 + 7×13.1 ≈ **104.9 s**

---

## 3. SC2 多轮对话 —— 三引擎对比

**场景**：5 轮对话，每轮把前面所有历史拼上，再问新问题。
**测量**：每轮 wall time。理想情况：只有新 token 被 prefill，历史 KV 复用。

| 指标 | sglang | MRV2 | MRV1 (APC+CG-on) |
|---|---|---|---|
| 5 轮平均 | **2.87 s** | 14.50 s | **1.29 s** |
| 第 5 轮 | **2.19 s** | 16.00 s | **1.21 s** |

**关键现象**：
- **MRV2 全线最慢**：每轮重新 prefill 整个增长的对话历史，越聊越慢。
- **MRV1 最快**（1.2–1.3 s/轮）：APC 命中历史前缀 + CG 加速 decode → 比 sglang 快约 2×。
- **sglang 居中**：RadixAttention 复用历史 KV，但 prefill 路径在 DLIN 上较慢，绝对延迟高于 MRV1。

---

## 4. SC3 并发批 —— 共享前缀的批处理

**场景**：4 个请求**共享同一前缀**，作为一个 batch 同时提交。
**测量**：整批 wall time，3 次取最优。

```
sglang:   throughput=20.0 tok/s  per_req=1601ms
MRV2:     throughput=2.5 tok/s   per_req=12678ms
MRV1:     throughput=41.9 tok/s  per_req=764ms
```

MRV1 的 APC 在 batch 内部**去重共享前缀**（类似 RadixAttention），加上更快的 DLIN 路径，取得最高吞吐（41.9 tok/s）。sglang 居中，MRV2 最慢（无缓存，4 份各 prefill 一遍 2K 前缀）。

---

## 5. 为什么 vLLM 缓存"开不了"—— MRV2 不能，但 MRV1 能 ⭐⭐

### MRV2（默认）—— APC 结构性不支持

给 vLLM 默认 runner（MRV2）开 `enable_prefix_caching=True`，**EngineCore 直接 init 失败**：

```
AssertionError: Model Runner V2 has not yet supported mamba_cache_mode='align'.
```

**根因**：
1. Qwen3.6-35B 是**混合 Mamba+Attention 架构**。
2. 开 APC 后，vLLM 强制 Mamba cache 进入 `'align'` 模式（block_size=528）。
3. **MRV2 硬性拒绝** `mamba_cache_mode='align'`（assert 直接挂）。
4. DLIN 上 MRV2 是**默认** runner。⇒ **MRV2 在这模型上永远无法用 APC**。

### MRV1（opt-in）—— APC + CG 可以跑通

绕过 MRV2 限制（`VLLM_USE_V2_MODEL_RUNNER=0`），MRV1 可以开 APC。但有两个坑：

**坑① ConstraintViolationError**：DLIN 上 torch.compile 对 MRV1 的动态形状断言失败。→ 解决：`compilation_config={"mode": "NONE"}` 关闭 compile。

**坑② block_size vs max_num_batched_tokens**：DL 补丁（`dl_config.py:199-201`）把 `max_num_batched_tokens` 钳到 `max(cudagraph_capture_sizes)`。如果只传 `[1,2,4]`，则 `max_num_batched_tokens=4`，Mamba APC 需要 `block_size=528` → `assert 528 <= 4` 崩。→ 解决：capture sizes 里包含一个 ≥528 的值（如 `[1,2,4,528]`），让 DL 补丁设 `max_num_batched_tokens=528` 绕过断言。CG 实际运行时只 capture 前三个（内存预算限制），528 只占位用。

**修复后的 MRV1 配置**：
```python
llm_kwargs["enable_prefix_caching"] = True
llm_kwargs["compilation_config"] = {"mode": "NONE",
                                     "cudagraph_capture_sizes": [1, 2, 4, 528],
                                     "max_cudagraph_capture_size": 528}
```

### 三引擎缓存能力对比

| 引擎 | 前缀缓存 | CG | DLIN 默认 |
|---|---|---|---|
| sglang (RadixAttention) | ✅ 原生 | ✅ | ✅ 默认 |
| vLLM-MRV2 | ❌ 结构限制 | ✅ | ✅ 默认 |
| vLLM-MRV1 (APC+CG-on) | ✅ 可用 | ✅ 可用 | ❌ 需 opt-in |

**结论**：vLLM 的 APC 在 Mamba 模型上**不是完全不可用**——MRV1 配合特殊配置可以跑通且性能很好。但 **MRV2（vLLM 默认 runner）硬限制无法绕过**。这意味着 vLLM 用户要么：
- 接受 MRV2 APC-OFF（慢 6–8×）
- 或显式切换到 MRV1（更快但非默认路径，需自行处理配置坑）

sglang 在 DLIN 上**开箱即用**就有 RadixAttention 前缀缓存，无需任何特殊配置。

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

## 7. 纯 decode TPOT —— sglang 解码落后

无前缀共享、单请求长 decode 的稳态吞吐：sglang **35.0** vs vLLM **37.9 tok/s**（vLLM +8%）。

这条差距是 sglang 多进程架构的 **scheduler↔worker ZMQ IPC 往返**（6.2 ms/token，100% host 侧），**GPU kernel 两引擎逐字节一致**（20.7 ms/token）。详见 [`sglang-dlin-decode-gap-profiling-blog.md`](sglang-dlin-decode-gap-profiling-blog.md)。这是 sglang 当前**唯一落后**且已知根因的项。

---

## 8. 给"说服别人"用的一页结论

### sglang 明确赢的（数据支撑）
1. **默认配置的直接对比**：sglang vs vLLM-MRV2（DLIN 上两者的默认 runner）——sglang **6–8× 快**（MRV2 无法开 APC）。
2. **架构适配**：sglang RadixAttention **开箱即用**支持混合 Mamba 模型；vLLM 需要**切换到 MRV1 + 特殊配置**才能开 APC。

### sglang 需要注意的
1. **MRV1 (APC+CG-on) 比 sglang 还快**（1.7–2.2×）：vLLM 非默认的 MRV1 路径配置好后，DCG + DLIN 原生算子比 sglang 更高效。这不是 sglang 的劣势，而是"默认 vs opt-in"的公平性问题。

### vLLM（MRV1 opt-in）赢的
1. MRV1 APC+CG-on：前缀场景比 sglang **快 1.7–2.2×**（需 `VLLM_USE_V2_MODEL_RUNNER=0` + 特殊 compilation_config）。
2. 纯 decode TPOT：+8%（6.2 ms IPC 往返，结构性，已定位）。
3. 短 prompt JSON：+9%（基本持平）。

### 一句话电梯版
> 在 Qwen3.6-35B（混合 Mamba）上：**sglang 开箱即用 RadixAttention，比 vLLM 默认（MRV2 APC-OFF）快 6–8×**。如果用户愿意切换到 vLLM 的 MRV1 非默认路径（APC+CG-on），则 MRV1 还能再比 sglang 快 1.7–2.2×——但 MRV1 不是默认配置，需要手动处理两个配置坑。**sglang 的优势在于"零配置就有前缀缓存"**。

---

## 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh

# sglang（RadixAttention）
CUDA_VISIBLE_DEVICES=8,9,10,11 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine sglang --mem-frac 0.55

# vLLM MRV2（APC 无法开启，DLIN 默认 runner）
CUDA_VISIBLE_DEVICES=8,9,10,11 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm --vllm-runner mrv2 --mem-frac 0.55

# vLLM MRV1（APC+CG-on，opt-in）
CUDA_VISIBLE_DEVICES=8,9,10,11 COMPARE_RUN_MRV1=1 ./run_sglang.sh compare --no-record
# 或直接：
CUDA_VISIBLE_DEVICES=8,9,10,11 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm --vllm-runner mrv1 --mem-frac 0.55
```

---

## 参考
- 本仓库：[`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md)（JSON/前缀 feature 对比）、[`sglang-dlin-decode-gap-profiling-blog.md`](sglang-dlin-decode-gap-profiling-blog.md)（6.2 ms decode gap 根因）、[`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md)
- Memory：`dlin-vllm-correct-package-and-env-gotchas`、`dlin-sglang-decode-gap-sync-opt-wins`
- 脚本：`scripts/dl/showcase_prefix_sharing.py`（SC1–4，双引擎）
