# sglang vs vLLM 展示——严谨性分析（胜出是否公平？）

> 配套文档：[`sglang-vs-vllm-new-scenarios.md`](sglang-vs-vllm-new-scenarios.md)
>（SC5/SC7/SC8/SC9/SC10 结果）和 [`sglang-vs-vllm-showcase-dlin.md`](sglang-vs-vllm-showcase-dlin.md)
>（SC1–SC4）。本文档**压力测试每个新场景的严谨性**——每个 sglang-vs-vLLM 对比是否公平，或者是否有混淆因素在夸大差距？——并给出每个结论背后的证据链（"证据路径"）。
>
> 模型：Qwen3.6-35B-A3B-FP8，TP4，DLIN。Commit `055e35d24f`（r017：MRV1 APC+CG-on 修复加入三引擎比较）。
> 诊断探针于 2026-07-24 添加；MRV1 修复于 2026-07-27。

## TL;DR —— 各场景结论

| 场景 | sglang vs MRV2 | MRV1 加入后的新视角 |
|----------|----------------|----------------------|
| **SC1** 前缀共享 | sglang warm **6.5×**（cold→warm 10.4×） | MRV1 (APC+CG-on) **1206 ms** < sglang 2010 ms（MRV1 1.67× > sglang） |
| **SC2** 多轮对话 | sglang turn5 **7.3×** | MRV1 **1205 ms** < sglang 2192 ms（MRV1 1.82× > sglang） |
| **SC3** 并发批处理 | sglang **8.0×** | MRV1 **41.9 tok/s** > sglang 20.0 tok/s（MRV1 2.1× > sglang） |
| **SC4** 短 JSON | vLLM **+9%** | — |
| **SC5** 多用户分叉 | sglang **8.22×** | MRV1 未测试 |
| **SC7** 长 RAG | sglang **16.25×** | MRV1 未测试 |
| **SC10** 共享系统提示词 | sglang **7.05×** | MRV1 未测试 |
| **SC11** 在线并发 | sglang **9.16×** | MRV1 未测试 |
| **SC8** best-of-N | sglang **5.84×** | MRV1 未测试 |
| **SC9** 纯解码 | vLLM **1.30×** | MRV1 未测试 |

**核心变化（2026-07-27）**：之前声称"vLLM APC 结构性不支持"已被推翻。**MRV1 配合特殊配置可以开 APC+CG-on，且性能超过 sglang 1.7–2.2×**。但 MRV2（vLLM 默认 runner）仍然 APC-OFF。所以公平性结论需要分两层：
- **sglang vs MRV2（默认）**：6–8× 领先，严谨。MRV2 APC-OFF 是结构性的，无法绕过。
- **sglang vs MRV1（opt-in）**：sglang 落后 1.7–2.2×。MRV1 需要 `VLLM_USE_V2_MODEL_RUNNER=0` + 特殊 compilation_config，非默认路径。

以下数据为 **sglang vs MRV2（DLIN 默认配置）** 的对比。MRV1 的发现改变了"APC 完全不可用"的结论，但不影响 sglang vs 默认 vLLM 的公平性。

---

## 1. 元严谨性：比较*框架*是否公平？

在逐个场景分析之前，先回答三个框架层面的问题。

### 1a. "sglang RadixAttention-开 vs vLLM APC-关"——这是公平对决吗？

**分引擎回答：**
- **sglang vs MRV2（默认）**：**是的，公平。** MRV2 结构性拒绝 `mamba_cache_mode='align'`（"Model Runner V2 has not yet supported mamba_cache_mode='align'"），APC-OFF 是唯一可工作配置。
- **sglang vs MRV1（opt-in）**：**MRV1 可以开 APC。** 2026-07-27 修复后，MRV1 + `compilation_config={"mode":"NONE","cudagraph_capture_sizes":[1,2,4,528]}` 可以同时开 APC + CG，且性能超过 sglang 1.7–2.2×。但 MRV1 不是 vLLM 默认 runner（需设置 `VLLM_USE_V2_MODEL_RUNNER=0`），也不是 DLIN 推荐路径。

**核心含义**：vLLM 的 APC 在 Mamba 模型上**不是结构性不可用**——只是 MRV2 路径堵死，MRV1 配合配置可以跑通。这意味着：
1. **sglang vs 默认 vLLM（MRV2）**：6–8× 领先，这个胜出是真实的。
2. **sglang vs MRV1（opt-in）**：sglang 落后——MRV1 有更快的 DLIN 原生路径。
3. sglang 的独特价值在于**零配置就有前缀缓存**，而 MRV1 需要用户了解并处理两个配置坑。

**重要边界：** 在 vLLM APC 可工作的稠密模型上，MRV2 也能开 APC，差距会缩小。

### 1b. 引擎配置是否同等最优？

三引擎使用各自的 DLIN 配置：

| 参数 | sglang | vLLM-MRV2 | vLLM-MRV1 (opt-in) |
|---|---|---|---|
| TP / dtype / FP8 | TP4, bf16, FP8（Q2 GEMM） | TP4, bf16, FP8（model） | TP4, bf16, FP8（model） |
| 内存比例 | 0.55 | 0.55 | 0.55 |
| cuda graph | 开，max_bs_decode=4 | 开，capture [1,2,4] | 开，capture [1,2,4,528] |
| 最大批处理序列数 | max_running_requests=4 | max_num_seqs=4 | max_num_seqs=4 |
| 前缀缓存 | RadixAttention **开** | APC **关**（结构性原因） | APC **开** |
| 编译配置 | 默认 | 默认 | mode=NONE（避免 DLIN torch.compile 崩溃） |

**注意**：MRV1 的 `mode=NONE` 关闭了 torch.compile，但这不影响性能——DLIN 上 compile 路径本身也有 bug（ConstraintViolationError）。MRV1 的 CG 仍然有效（`cudagraph_capture_sizes` 含 `[1,2,4]`，528 只是为了抬高 `max_num_batched_tokens` 绕过断言）。

三引擎的可比性：sglang vs MRV2 是公平的（两者都是 DLIN 默认路径）。MRV1 虽然更快，但需要非默认配置和两个工作区，不是"开箱即用"的对比。

### 1c. 指标是否在衡量我们声称的内容？

- **比率**（sglang 时间 / vLLM 时间，同一框架）是有效的信号——两个引擎支付相同的离线 `generate` 每调用开销，因此在比率中抵消。
- **绝对 tok/s**（例如 SC7 中 sglang 13 tok/s）*不是*服务器吞吐量——它们受离线开销限制（针对 24 token 的短解码）。sglang 的在线解码约为 ~35 tok/s。因此请阅读**比率**，而非绝对值。

### 1d. 引擎是否产生可比较的 OUTPUT？（贪心一致性探针）

速度比较衡量 token/秒，但从未检查 sglang 和 vLLM 在 temperature=0 时是否发出*相同*的 token。它们运行不同的 FP8 路径（sglang fused-MoE/fa3/GDN vs vLLM `_dl_C`），因此贪心输出可能发散——而如果确实发散，不同的 token 数 / EOS 点会使速度比较变成苹果对橘子的比较。探针：`scripts/dl/greedy_agreement.py`，6 个工作负载提示词（真实的 SC1/SC7/SC6/SC9 形状），temp=0，48 tok，`ignore_eos`，FUSED_MAX_M=2048：

| 提示词形状 | 一致性 | 解读 |
|---|---|---|
| **SC9** 短提示词（~14 tok），纯解码 | **48/48 完全一致** | 短 prefill → FP8 漂移可忽略 → 贪心保持锁定 |
| **SC1 / SC7 / SC6** 长/前缀共享 prefill（~2K tok） | **在 token 0–12 处分叉** | 2K token prefill 累积 FP8 漂移 → 贪心早期分叉 |

双方输出**都是连贯且事实上正确的**（例如都回答"6144 TFLOPS"、"每 QUAD 192 TFLOPS"）；都不是垃圾。发散是**良性的跨引擎 FP8 数值漂移**，被贪心解码放大——这是 FP8 推理在不同内核实现下的预期行为，**不是**任一引擎的**bug**。

**对速度比较的影响：** 仍然**有效**——两个引擎发出相同的 token 数（`ignore_eos`），tok/s 与 token 一致性无关，且解码每 token 成本大致恒定。但输出**质量在长 prefill 场景下不等价**：我们声称" sglang 更快"，而非" sglang 产生相同的文本"。这正是 F1（对 SC1/SC2/SC3 使用 `ignore_eos=True`）的必要性：由于输出发散，必须锁定 token 数，否则不同引擎的不同 EOS 点会扭曲吞吐量（F2 风险）。经验上，此模型会进入 `<think>` 且不会提前 EOS（`nat_ntok` = 48/48），因此 F1 的数值影响很小（SC3 20.2→19.8 tok/s）——它是正确性守卫，而非数字修正。

---

## 2. 关键实验：SC6 原始 prefill 对比

最重要的严谨性问题：**胜出来自 RadixAttention 缓存，还是也来自 sglang 更快的原始 prefill 内核？** 如果 sglang 只是 prefill 更快（相同 `_dl_C` 内核，但编排更好），那么"缓存"的说法就错了。

**SC6** 回答这个问题：N 个**独特** ~1.3K token 提示词（不同前缀 ⇒ RadixAttention *无法*命中），+ 4 token 解码。两个引擎都完全 prefill 每个提示词。不同的提示词每次调用都不同，击败了跨调用缓存。

| SC6 原始 prefill（无缓存） | 速率 |
|---|---|
| **sglang** | **49 tok/s**（1337 tok × 8，219.8 秒） |
| **vLLM-MRV2** | **76 tok/s**（1337 tok × 8，140.0 秒） |
| → **vLLM 在原始 prefill 上快 1.55×** | |

**结论：** sglang 的原始 prefill **并不**比 vLLM 快——反而更慢。因此 SC5/SC7/SC8/SC10 的胜出**不能**用原始 prefill 速度解释；它们必须来自 **RadixAttention 缓存**（sglang 跳过共享前缀的 prefill；vLLM，APC 关闭，每次都支付 prefill）。这是干净的归因。（推论：在*全部独特*提示词的工作负载上，vLLM 胜出——SC6 *就是*那个工作负载，vLLM 快 1.55×。）

> 附注：sglang 的 49 tok/s 原始 prefill 慢得可疑（两个引擎共享 `_dl_C`）。可能是 M=1 情况下 `chunked_prefill_size=512` 的开销。与缓存的胜出无关——当 sglang *确实*缓存时，它只 prefill 微小的增量后缀（~10–50 tokens），因此慢速原始 prefill 速率永远不会成为瓶颈。

---

## 3. 逐个场景严谨性分析

### SC1 — 前缀共享 — 三引擎对比
- **机制：** 8 个请求共享 ~2K token 前缀。cold = 首次完整 prefill + JIT；warm = 7 次缓存命中。
- **结果（r017）**：
  | 引擎 | cold | warm | speedup | vs sglang |
  |---|---|---|---|---|
  | sglang | 20876 ms | **2010 ms** | 10.4× | — |
  | MRV2 | 13178 ms | 13058 ms | 1.01× | sglang 6.5× 快 |
  | MRV1 (APC+CG-on) | **1628 ms** | **1206 ms** | 1.35× | **MRV1 1.67× 快** |
- **分析**：sglang vs MRV2 的 6.5× 胜出来自缓存（MRV2 APC-OFF），严谨。MRV1 的 1.67× 优势来自更快的 DLIN 原生路径（无多进程 IPC）——这与"缓存 vs 原始速度"无关，而是引擎架构差异。MRV1 的 cold→warm 加速比仅 1.35×，说明 APC 虽然生效但不如 RadixAttention 的 token 级粒度高效。
- **⚠️ cold→warm 加速比受 JIT 污染（F5）**：sglang 的 10.4× 含 JIT 首次命中。**使用 warm 延迟作为干净指标。**

### SC2 — 多轮对话 — 三引擎对比
- **机制：** 5 轮对话，每轮追加到历史。RadixAttention/APC 缓存增长的历史；MRV2 每轮重新 prefill 整个历史。
- **结果（r017）**：
  | 指标 | sglang | MRV2 | MRV1 |
  |---|---|---|---|
  | 5 轮平均 | **2872 ms** | 14504 ms | **1290 ms** |
  | 第 5 轮 | **2192 ms** | 15998 ms | **1205 ms** |
- **分析**：sglang vs MRV2 的 7.3× 胜出来自缓存（MRV2 APC-OFF），严谨。MRV1 比 sglang 快 1.82×（更快的 DLIN 路径）。

### SC3 — 并发批处理 — 三引擎对比
- **机制：** 4 个前缀共享的请求作为一个批处理提交。
- **结果（r017）**：
  | 指标 | sglang | MRV2 | MRV1 |
  |---|---|---|---|
  | 吞吐 | **20.0 tok/s** | 2.5 tok/s | **41.9 tok/s** |
  | 每请求 | **1601 ms** | 12678 ms | **764 ms** |
- **分析**：sglang vs MRV2 的 8.0× 胜出来自缓存，严谨。MRV1 2.1× > sglang。MRV1 的 APC 在 batch 内部去重共享前缀（类似 RadixAttention），加上更快的 decode 路径，取得最高吞吐。

### SC4 — 短 JSON（vLLM +9%）— ✅ 严谨（内置有效性检查）
- **机制：** 使用 schema 提取到 JSON；这是唯一解析输出并记录 `valid` 的场景。两个引擎都产生**有效的 JSON**。vLLM 在此短提示词、解码轻、无共享前缀的提示词上快约 ~9%——这是 vLLM 的公平胜出，与 SC9 风格相同（解码 IPC，无缓存结构可利用）。

### SC5 — 多用户分叉（sglang 8.22×）— ✅ 严谨
- **机制：** 2 个用户 × 4 轮对话，共享一个根节点，交错执行。sglang 缓存根节点 + 每个用户增长的分支；每轮只扩展增量后缀。vLLM（APC 关闭）每轮重新 prefill 整个增长的对话历史。
- **公平性：** 相同模型/GPU/提示词/解码(24)/temp(0)。✓
- **混淆因素检查（SC6）：** 8.22× *不是*原始 prefill 速度（sglang 那里更慢）——这是 vLLM 每轮重新 prefill 的成本。确认是缓存。✓
- **缓存真实性检查：** sglang 的用时随轮次减少（15519→13215 ms），与 radix 树预热一致。✓
- **结论：严谨。** 真实的工作负载（多租户），归因正确。

### SC7 — 长 RAG 吞吐量（sglang 16.25×）— ✅ 严谨
- **机制：** ~2K token 文档 × 8 个顺序查询。sglang prefill 文档一次；vLLM 重新 prefill ~2K tokens × 8 查询 × 多轮。
- **混淆因素检查（SC6）：** vLLM 的慢速 prefill（76 tok/s ⇒ 每 2K 重新 prefill 约 ~28 秒）是它的*原始*速率，而非配置削弱（SC6 在独特提示词上测量了它）。sglang 通过缓存避免它。✓
- **结论：严谨。** 大比率反映了大前缀 × 多查询——正是缓存最能发挥作用的地方。

### SC10 — 共享系统提示词（sglang 7.05×）— ✅ 严谨
- 与 SC7 相同的机制（缓存），但前缀更短（0.9K）× 更多租户（12）。SC6 确认归因。✓ **严谨。**

### SC11 — 在线并发（sglang 9.16×）— ✅ 严谨（FA2 解除阻塞）
- **机制：** 12 个租户在共享 ~0.9K 系统提示词后**并发**触发（一个批处理），每个租户有不同长度的查询（不等长 `extend_lens` → FA2 varlen 路径）。生产形状的多租户并发修复 `0fe8c7cc86` 解除了阻塞（修复前不等长并发批处理 OOB/SIGSEGV）。与 SC10（相同形状，**顺序**执行）的区别在于并发性；框架将并发限制在 `max_running_requests=4` → 12 个租户以 4 个一波进行处理。
- **结果（r016）：** sglang **17.4 tok/s** vs vLLM **1.9** → **9.16×**。
- **归因：** 与 SC5–SC10 相同——RadixAttention 缓存（vLLM APC 关闭，每轮重新 prefill 共享系统提示词 ×12；SC6 排除了原始 prefill 速度），**被并发放大**（9.16× > SC10 顺序 7.05×——并发解码批处理有助于 sglang）。FA2 修复是*使能者*，而非胜出来源。
- **混淆因素检查：** 缓存驱动（✓ SC6）；没有每个场景的失败；不等长并发路径本身在 FA2 e2e 测试中验证过（离线批处理 + 在线服务器）。**严谨。**

### SC8 — best-of-N 并行采样（sglang 5.84×）— ⚠️ 混合（两个因素）
- **混淆因素：** SC8 框架运行 warmup + 3 个测量轮次，使用**相同的** ~0.9K 提示词。sglang 的 RadixAttention 在 warmup 后缓存了该提示词 ⇒ 测量轮次跳过 prefill（缓存命中）。vLLM（APC 关闭）每轮重新 prefill。
- **探针（SC8b）：** 与 SC8 相同，但每次调用使用**独特**提示词 ⇒ 无跨调用缓存。测量结果：

  | best-of-N（n=4，temp=0.7） | sglang | vLLM |
  |---|---|---|
  | SC8 warm（相同提示词复用 → sglang 缓存） | **22.2 tok/s** | 3.8 tok/s |
  | SC8b cold（独特提示词/调用 → 无缓存） | **5.7 tok/s** | 2.8 tok/s |

- **SC8 5.84×（22.2 / 3.8）的分解：**
  - **跨调用缓存 ≈ 3.9×** — sglang warm（22.2）vs sglang cold（5.7）。这是 RLHF 拒绝采样循环的优势（提示词被多次重新生成；sglang 缓存它，vLLM 不能）。
  - **单调用 best-of-N 优势 ≈ 2.0×** — sglang cold（5.7）vs vLLM cold（2.8）。sglang 在 DLIN 上处理 n=4 并行采样比 vLLM 更好。
- **vLLM 的 n=4 病态：** vLLM 的 n=4 best-of-N 聚合速率是 **2.8 tok/s**——而其 n=1 单流解码是 **39.6 tok/s**（SC9）。每序列约 ~0.7 tok/s，比 n=1 慢约 50×。vLLM 显然在 DLIN 上**没有**有效批处理/捕获 n=4 解码（CG capture 是 `[1,2,4]`，所以 batch=4 *应该*被捕获——很可能是个 best-of-N 代码路径问题）。因此 SC8 的部分胜出可能是可修复的 vLLM n=4 弱点。sglang 的 n=4（5.7）也远低于人们期望的 ~4× 单流（≈120）的批处理解码——所以两个引擎的 n=4 批处理都不好，但 sglang 不那么差。
- **结论：** **两个因素都是 sglang 的真实胜出**，但它们衡量的是不同的工作负载。SC8（5.84×）是**重复 best-of-N / RLHF 循环**数字（缓存主导）。SC8b（~2×）是**单调用 best-of-N** 数字。修复：将 SC8 重新标记为"重复 best-of-N（RLHF 循环）"，并将 SC8b 作为单调用数字；标记 vLLM 的 n=4 病态，以免单调用 2× 被解读为基本的解码优势。

### SC9 — 纯长解码（vLLM 1.30×）— ⚠️ 方向严谨，幅度存疑
- **机制：** ~14 token 提示词 + 128 token 单流贪心解码。prefill 可忽略 ⇒ 解码受限。vLLM 的 DLIN 解码 IPC 优势胜出。
- **方向稳健：** vLLM 39.6 > sglang 30.5 tok/s——vLLM 此处确实解码更快。✓
- **幅度注意事项：** sglang 的 30.5 tok/s **低于其约 ~35 tok/s 的最佳值**（在调优的解码配置中测量，参见 `sglang-dlin-decode-gap-debug-blog`）。展示配置（mem 0.55，max_running_requests=4，短 warmup）未达标。在 sglang 的 35 最佳值下，差距为 39.6/35 = **1.13×**，而非 1.30×。因此" vLLM 1.30×"是一个**上界**；真实的解码差距约为 ~1.1–1.3×。
- **结论：** 严谨地确定 vLLM 解码胜出；将幅度报告为范围。

---

## 4. 证据路径（证据链）

对于每个结论，可以重新推导的证据：

1. **"sglang vs MRV2 胜出来自缓存，而非原始 prefill"** ← SC6（独特提示词原始 prefill：sglang 49 < vLLM 76 tok/s）。由于 sglang 原始*更慢*，任何 sglang vs MRV2 胜出必定来自缓存。重现：`compare --scenarios SC6`。
2. **"SC5/SC7/SC10 是干净的缓存胜出"** ← SC6（#1）+ 场景自身的递减轮次（缓存预热）+ 机制（共享前缀，vLLM 重新 prefill）。重现：`compare --scenarios SC5,SC7,SC10`。
3. **"SC8 = 缓存（3.9×）+ 单调用 best-of-N 优势（2×）"** ← SC8（相同提示词复用）：sglang 22.2 vs vLLM 3.8。SC8b（独特提示词/调用）：sglang 5.7 vs vLLM 2.8。sglang-warm/sglang-cold = 22.2/5.7 = 3.9×（缓存因子）；sglang-cold/vLLM-cold = 5.7/2.8 = 2.0×（单调用 best-of-N 因子）。vLLM 的 n=4（2.8 tok/s）vs 其 n=1（39.6，SC9）标记了 n=4 病态。重现：`compare --scenarios SC8,SC8B`。
4. **"SC9：vLLM 解码胜出，幅度 ~1.1–1.3×"** ← SC9（vLLM 39.6 > sglang 30.5）+ 已知 sglang 解码最佳值（~35，来自解码差距工作）。
5. **"MRV2 APC-OFF 是 MRV2 的唯一选项"** ← MRV2 硬拒绝 `mamba_cache_mode='align'`。**但 MRV1 可以开 APC+CG-on**（2026-07-27 r017 修复）。参见 `showcase_prefix_sharing.py` 的 MRV1 分支配置。

---

## 5. 诚实总结与修复

**sglang vs MRV2（默认）——坚如磐石：** SC1/SC2/SC3 的 6–8× 胜出是真实的缓存胜出（MRV2 结构性 APC-OFF），SC5/SC7/SC10/SC11 的 7–16× 同样。SC4/SC9 是 vLLM 的公平胜出。

**MRV1 发现（2026-07-27，r017）：** MRV1 (APC+CG-on) 在 SC1–SC3 上比 sglang 快 1.7–2.2×。这**不影响** sglang vs MRV2（默认）的对比严谨性，但修正了"vLLM APC 完全不可用"的结论——MRV2 不能，但 MRV1 可以用。sglang 的独特价值是"零配置就有前缀缓存"。

**本轮应用的正确性修复（F1/F4）：**
- **F1：** SC1/SC2/SC3 现在强制 `ignore_eos=True`（SC3 通过 `generate_batch`），使两个引擎发出相同的 token 数——这是必需的，因为两个引擎的贪心输出在长 prefill 上发散（§1d；否则不同的 EOS 点会扭曲吞吐量）。数值影响很小（SC3 20.2→19.8 tok/s）：此模型进入 `<think>` 且不会提前 EOS，因此 F1 是正确性守卫，而非数字修正。
- **F4：** 每个场景的失败现在会发出明确的 `SCx_status=fail` 标记（持久化到 JSON 存储 + CSV），而非静默缺失。

**输出等价性注意事项（新增，§1d）：** sglang 和 vLLM 在长 prefill 上**不会**产生 token 一致的输出（跨引擎 FP8 漂移，被贪心解码放大；仅在短 prefill 纯解码上一致——SC9）。速度比较仍然有效，但声称的是" sglang 更快"，而非" sglang 产生相同的文本。"

**需要修复：**
- **SC8：** 重新标记为"重复 best-of-N（RLHF 拒绝采样循环）"——5.84× 受缓存主导。将 **SC8b（~2×，单调用）** 作为单 best-of-N 数字呈现，并**标记 vLLM 的 n=4 病态**（2.8 tok/s vs 39.6 n=1），以免单调用 2× 被误解为 sglang 的基本解码优势——它部分是一个可能可修复的 vLLM n=4 弱点。
- **SC9：** 将幅度陈述为 ~1.1–1.3×（sglang 此处解码低于最佳值）。

**不是严谨性问题，但值得注意：**
- sglang 的原始 prefill（49 tok/s）比 vLLM 的（76）慢——可能是 M=1 情况下 `chunked_prefill_size=512` 的开销。不影响缓存的胜出（缓存的后缀很小），但这是 sglang prefill 的真实低效问题，值得单独查看（增大分块大小 / 融合 M=1 prefill）。

**什么会改变胜出结果：**
- MRV1 APC+CG-on **已经**改变了部分结论（vLLM 也可以缓存并跑得比 sglang 快），但 MRV1 不是默认路径。
- 稠密模型上 MRV2 也能开 APC，sglang vs MRV2 的 6–8× 差距将缩小到解码/prefill 差异。
- MRV1 的 DLIN 原生路径比 sglang 更高效（无多进程 IPC），这是 sglang 架构层面的优化方向。
