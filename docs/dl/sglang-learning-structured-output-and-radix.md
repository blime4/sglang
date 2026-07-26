# SGLang 学习路线：结构化输出 & 前缀共享（RadixAttention）

> 学习目标：搞懂 SGLang 相对 vLLM 的两块"结构性优势"的原理、论文、实现。
> - **方向 A**：结构化输出（JSON/regex）—— Compressed FSM + XGrammar
> - **方向 B**：前缀共享 / 多轮 / Agent —— RadixAttention + Cache-Aware Router
>
> 每个方向都给：①核心概念速览 ②论文清单（按难度）③Blog 清单 ④**本仓库代码入口** ⑤推荐阅读顺序。
> 难度标记：🟢入门 🟡进阶 🔴前沿。⭐ = 必读。
> 所有"本仓库代码入口"都基于当前 `dl-main` 分支的真实路径（含行号），可点开直接读。

---

## 0. 前置知识（两块共用）

学这两块之前，建议先掌握：

| 概念 | 为什么需要 | 推荐速读 |
|---|---|---|
| 自回归解码 / KV cache | 没有它就理解不了"为什么要复用 KV" | vLLM PagedAttention 论文 §2 |
| Logits + Sampling | 结构化输出的本质是"改 logits" | 任意 LLM 推理综述 |
| PagedAttention | RadixAttention 的直接前置 | vLLM 论文 (SOSP'23) |
| 正则表达式 / 有限状态机 (FSM) | 结构化输出的核心数据结构 | Outlines 论文 §3 |
| 上下文无关文法 (CFG) / 下推自动机 (PDA) | XGrammar 的核心数据结构 | XGrammar blog §Background |

---

# 方向 A：结构化输出（Compressed FSM + XGrammar）

## A.1 核心概念速览

**问题**：让 LLM 100% 输出合法 JSON / 正则匹配的串。例如 `{"name": "...", "age": <int>}`。

**本质技术 = 约束解码（Constrained Decoding）**：每生成一个 token 前，把"会导致非法输出"的 token 在 logits 里 mask 掉，采样就只能在合法 token 里选。

```
LLM logits  →  [grammar/regex 算出合法 token 集合]  →  logits[mask 非法]=-inf  →  sample
```

这里有两个独立的核心难点：

### 难点 1：如何高效算出"合法 token 集合"？
- **朴素 FSM（Outlines 的做法）**：把 JSON Schema → 正则 → DFA，对每个 state 预计算"哪些 token 合法"。问题：词表 128k × state 数 = 巨大 mask 矩阵；CFG 无法预计算（状态无限）。
- **XGrammar 的关键洞察**：把 token 分成两类 ——
  - **Context-independent token**（>99%）：只看当前 PDA 位置就能判定，可**预编译进 adaptive token mask cache**；
  - **Context-dependent token**（<1%）：必须运行时查栈。
  运行时只查那 1%，所以 mask 生成近乎零开销。→ 这就是"比 Outlines 快 10×"的根因。

### 难点 2：如何让约束解码不拖慢生成？
- **朴素问题**：FSM 一个 state 只能转移一个 token，所以只能逐 token 解码 → 慢。
- **Compressed FSM + Jump-Forward（SGLang 原创）**：分析 regex 的 FSM，把"确定性单边路径"（singular paths）压缩成长串。遇到这种确定段，**直接 prefill 多个 token，一步跳到下一个分叉点**（jump-forward）。例如知道接下来一定是 `{\n  "name":`，就一把 prefill 出去，不等逐 token 生成。
  - 关键：jump-forward 复用 KV cache（终止当前请求 → 入队新请求，RadixAttention 自动复用前面 token 的 KV），所以"提前填充"几乎免费。
- **XGrammar 的 overlap**：把 grammar 的 mask 计算和 GPU 推理重叠，CPU 不阻塞 GPU。

> **一句话区分**：Compressed FSM 是 SGLang 2024-02 提出的"加速正则约束"算法（基于 Outlines）；XGrammar 是 MLC 2024-11 提出的"用 CFG 代替 regex、且 mask 预编译"的引擎。SGLang v0.4 把 XGrammar 集成进来当 grammar backend，两者叠加 → JSON 解码比其他方案快最多 10×。

## A.2 论文清单（按难度）

| 难度 | 论文 | 读什么 |
|---|---|---|
| 🟢⭐ | **Outlines**: Willard & Louf, *Efficient Guided Generation for LLMs*, arXiv 2307.09770 (2023) | FSM 约束解码的奠基作。必读 —— 理解"regex → DFA → per-state token mask"整条链。 |
| 🟡⭐ | **SGLang**: Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv 2312.07104 (2023) | 第 3 节讲 Compressed FSM + jump-forward + frontend DSL（`gen`/`select`/`fork`）。 |
| 🟡⭐ | **XGrammar**: Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*, arXiv 2411.15100 (2024) | context-independent/dependent token 分类、adaptive token mask cache、PDA 优化。SGLang 现在默认用它。 |
| 🔴 | **XGrammar 2**: *Efficient Dynamic Structured Generation for Agentic Workloads*, arXiv 2601.04426 (2026) | 面向 agent 的动态结构化生成，已进 SGLang/vLLM/TRT-LLM。 |
| 🟡 | **GAR**: *Grammar-Augmented Generation*, arXiv 2310.04663 | 另一条 CFG 路线，对比阅读理解设计空间。 |
| 🟢 | **LMQL**: Beurer-Kellner et al., *Prompting Is Programming*, arXiv 2212.06094 | 早期声明式约束语言，SGLang 前端 DSL 的灵感来源之一。 |
| 🟢 | Synchromesh / Picard (2022/2023) | 更早的 SQL/代码约束生成，历史脉络。 |

## A.3 Blog 清单

| 优先级 | Blog | 要点 |
|---|---|---|
| ⭐⭐ | [Fast JSON Decoding with Compressed Finite State Machine](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/) (LMSYS, 2024-02) | jump-forward + compressed FSM 的图解，比 Outlines/vLLM 延迟 2×、吞吐 2.5×。配 GIF demo。 |
| ⭐⭐ | [XGrammar: Efficient Flexible Portable Structured Generation](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) (MLC, 2024-11) | CFG/PDA background、token 分类、overlap 流水线、e2e H100 benchmark（CFG 比竞品快 80×）。 |
| ⭐ | [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) §"Fast Structured Outputs" | XGrammar 集成 + `--grammar-backend xgrammar`，比其他开源方案快最多 10×。 |
| — | [XGrammar 2 blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) (2026-05) | 最新动态结构化生成。 |

## A.4 本仓库代码入口（真实路径，可点开读）

```text
python/sglang/srt/constrained/            # 约束解码总目录
├── base_grammar_backend.py               # 🟢 通用接口：GrammarBackend / GrammarObject
├── grammar_manager.py                    # 🟢 每个 req 一个 GrammarManager，负责每步 mask logits + jump-forward 触发
├── xgrammar_backend.py                   # 🟡⭐ XGrammar 适配层：XGrammarGrammarBackend + allocate_vocab_mask (line 59-230)
├── outlines_backend.py                   # 🟡 Outlines (FSM) 适配层
├── outlines_jump_forward.py              # 🟡⭐ Compressed FSM + jump-forward 的核心实现（SGLang 原创）
├── llguidance_backend.py                 #    另一个 grammar 后端选项
└── reasoner_grammar_backend.py           #    thinking 模型的 grammar 处理

sgl-kernel/csrc/grammar/                  # grammar 相关 CUDA kernel
sgl-kernel/python/sgl_kernel/grammar.py   # Python 暴露的 grammar op
```

**建议读码顺序**：
1. `base_grammar_backend.py`（看接口抽象）
2. `xgrammar_backend.py`（看 SGLang 怎么调 XGrammar）
3. `outlines_jump_forward.py`（看 compressed FSM + jump-forward 怎么实现"提前填充"）
4. `grammar_manager.py`（看 mask 在采样循环里怎么注入 —— 跟 scheduler 连起来）

## A.5 推荐学习路径（A 方向，约 1 周）

```
Day 1-2  🟢 Outlines 论文 + Compressed FSM blog        → 搞懂 FSM mask + jump-forward
Day 3    🟢 SGLang 论文 §3                              → 压缩 FSM 的形式化 + DSL
Day 4-5  🟡 XGrammar 论文 + blog                        → CFG/PDA + token 分类预编译
Day 6    🟡 读 outlines_jump_forward.py + xgrammar_backend.py
Day 7    🔬 实操：启动 --grammar-backend xgrammar，跑一个 JSON schema，对比 outlines
```

---

# 方向 B：前缀共享 / 多轮 / Agent（RadixAttention + Cache-Aware Router）

## B.1 核心概念速览

**问题**：多个请求共享相同前缀（system prompt、few-shot、多轮历史、self-consistency），重复算这些前缀的 KV 是浪费。

**RadixAttention 的核心 = 用 radix tree 管理 KV cache**：
- 把"token 序列 → KV cache 张量"的映射存在一棵 radix tree 里（节点是共享前缀，边是 token 子串）。
- 每个新请求来了，**自动做前缀匹配**，命中部分复用其 KV，未命中的再算。
- GPU 内存有限 → **LRU 淘汰**（递归淘汰叶子节点）。
- 配合 **cache-aware 调度**：优先调度"前缀命中多"的请求，提高命中率。
- 数据结构在 CPU 上，维护开销小，且兼容 continuous batching + paged attention。

> **RadixAttention vs vLLM prefix cache（APC）**：SGLang 是 **token 级 radix tree**（任意前缀自动发现）；vLLM 早期是 **block 级 hash 匹配**。两者后来趋同，但 radix tree 在复杂共享模式（多分支、多轮）下命中率更高。

**多机 / 多 DP worker 时的 Cache-Aware Router**（SGLang v0.4）：
- 问题：`--dp-size N` 时 round-robin 路由会让同一个前缀被打散到不同 worker，命中率塌掉。
- 解决：router 在本地维护一棵 worker 真实 radix tree 的**"近似树"**（通过 KV events 增量同步），**把请求路由到命中率最高的 worker**。
- 效果：命中率 20% → 75%，吞吐 +1.9×。
- 实现用 **Rust**（`sglang-router` / `sgl-router`），高并发低开销。

## B.2 论文清单（按难度）

| 难度 | 论文 | 读什么 |
|---|---|---|
| 🟢⭐ | **vLLM / PagedAttention**: Kwon et al., SOSP'23, arXiv 2309.06180 | paged KV cache 的奠基。RadixAttention 的直接前置（SGLang 也复用了它的 paged layout 思想）。 |
| 🟢⭐ | **SGLang**: Zheng et al., arXiv 2312.07104 | 第 4 节 RadixAttention：radix tree、LRU、cache-aware 调度的形式化。 |
| 🟡⭐ | **Preble**: Srivatsa et al., *Efficient Distributed Prompt Scheduling*, ICLR'25, arXiv 2407.00023 | **cache-aware router 的学术基础**。Global Prompt Tree + E2 调度器（联合优化"前缀复用"和"负载均衡"）。1.5–14.5× 延迟改善。SGLang router 的思想来源。 |
| 🟡 | **DistServe**: Zhong et al., OSDI'24, arXiv 2401.09670 | prefill/decode 分离；理解大规模前缀共享为什么需要分离架构。 |
| 🟡 | **Mooncake**: DeepSeek, arXiv 2407.00079 | KV-cache-centric 分离架构，生产级前缀共享。 |
| 🔴 | **CacheBlend / CacheGen**: arXiv 2310.07240 / 2405.16444 | KV cache 的压缩与跨请求复用进阶。 |
| 🟡 | **Prompt Cache**: Jin et al., NeurIPS'24, arXiv 2310.10167 | 把 prefix KV 当成可复用的"知识包"。 |
| 🟢 | **S-LoRA**: Sheng et al., arXiv 2311.03285 | 同一团队（RadixAttention 作者）的前作；Unified Paging（KV + adapter 统一分页），思想延续到 RadixAttention。 |

## B.3 Blog 清单

| 优先级 | Blog | 要点 |
|---|---|---|
| ⭐⭐ | [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) (LMSYS, 2024-01) | radix tree 9 步演化图（图4）、4 种共享模式、LRU、cache-aware 调度。最直观。 |
| ⭐⭐ | [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) §"Cache-Aware Load Balancer" | approximate tree、命中率 20%→75%、Rust router。 |
| ⭐ | [Preble blog (WukLab)](https://mlsys.wuklab.io/posts/preble/) | global prompt tree + E2 调度器的通俗版。 |
| — | [SGLang HiCache blog](https://www.lmsys.org/blog/2025-09-10-hicache/) (2025-09) | 分层 KV cache（GPU→CPU→存储），前缀共享的纵向延伸。 |
| — | [SGLang Multi-Level / HiCache 系列](https://www.lmsys.org/blog) | 进阶：长上下文前缀淘汰到 CPU/SSD。 |

## B.4 本仓库代码入口（真实路径，可点开读）

```text
# RadixAttention 主体
python/sglang/srt/mem_cache/
├── radix_cache.py          # 🟡⭐ 核心：RadixCache 类 (line 285)、TreeNode (222)
│                           #         match_prefix (358)、insert (418)、evict (558)
├── radix_cache_cpp.py      # 🔴 C++ radix tree 的 Python 绑定（生产路径）
├── cpp_radix_tree/         # 🔴 C++ radix tree 实现（高性能）
├── hiradix_cache.py        # 🔬 HiCache：GPU↔CPU↔存储 分层 radix cache
├── unified_radix_cache.py  # 🔬 统一 cache（新架构）
├── evict_policy.py         # 🟢 LRU 等淘汰策略
├── chunk_cache.py          # 🟢 chunked prefill 用的临时 cache
├── base_prefix_cache.py    # 🟢 PrefixCache 抽象接口
└── memory_pool.py          # 🟢 paged KV pool（配合 radix tree）

# Cache-Aware Router（Rust）
experimental/sgl-router/    # 🟡⭐ sgl-router（原 sglang-router）
├── src/lib.rs
├── src/main.rs
├── src/policies/
│   ├── kv_events/hash.rs        # 🟡⭐ 从 KV events 重建 worker 的"近似 radix tree"
│   ├── kv_events/discovery.rs   #     worker 发现 + tree 同步
│   ├── active_load.rs           #     负载感知
│   └── registry.rs              #     路由策略注册
├── benches/tree_lookup.rs       #     近似树查找性能基准
└── benches/policy_select.rs

# 调度器侧（命中检测 + cache-aware 调度）
python/sglang/srt/managers/scheduler.py   # 🔴 巨大文件；前缀匹配 / batch 组装在这里
```

**建议读码顺序**：
1. `radix_cache.py` 的 `RadixCache` / `TreeNode` / `match_prefix` / `insert` / `evict`（先看 Python 版，逻辑直观）
2. `base_prefix_cache.py`（看接口）
3. `experimental/sgl-router/src/policies/kv_events/hash.rs`（看 cache-aware router 怎么重建近似树）
4. （进阶）`cpp_radix_tree/` 和 `hiradix_cache.py`

## B.5 推荐学习路径（B 方向，约 1 周）

```
Day 1    🟢 vLLM/PagedAttention 论文 §2-3               → 先理解 paged KV
Day 2-3  🟢⭐ SGLang 论文 §4 + RadixAttention blog      → radix tree + LRU + cache-aware 调度
Day 4    🟢 读 radix_cache.py (RadixCache/TreeNode)
Day 5    🟡⭐ Preble 论文 + blog                        → 多机前缀调度、E2、近似树
Day 6    🟡 读 sgl-router/src/policies/kv_events/       → cache-aware router 的 Rust 实现
Day 7    🔬 实操：--dp-size 4 对比开/关 cache-aware router 的命中率
```

---

## 1. 实操实验清单（学完两边都做一遍）

| 实验 | 命令要点 | 验证什么 |
|---|---|---|
| JSON 结构化输出 | `--grammar-backend xgrammar` + OpenAI API `response_format={"type":"json_schema",...}` | XGrammar 生效 + 速度 |
| 对比 grammar 后端 | 分别用 `xgrammar` / `outlines` 跑同一 schema | 速度差异（验证 token 分类预编译） |
| 前缀缓存命中 | 跑多轮对话 / 同一 few-shot 多请求，看日志 `cache hit rate` | RadixAttention 自动命中 |
| Cache-aware router | `python -m sglang_router.launch_server ... --dp-size 4`，对比 naive round-robin | 命中率 20%→75% |

> ⚠️ 以上实验基于 NVIDIA GPU。本仓库跑在 **DLIN（DLIN）** 上时，注意 RadixAttention 本身是 CPU 侧逻辑、与硬件无关，但 XGrammar 的 mask kernel、paged KV pool 的实现需要 DLIN 适配 —— 结合 `docs/dl/` 下其他笔记对照看。

---

## 附录：完整引用

### 方向 A（结构化输出）
- Outlines: https://arxiv.org/abs/2307.09770
- SGLang: https://arxiv.org/abs/2312.07104
- XGrammar: https://arxiv.org/abs/2411.15100 ｜ repo https://github.com/mlc-ai/xgrammar
- XGrammar 2: https://arxiv.org/abs/2601.04426
- GAR: https://arxiv.org/abs/2310.04663
- LMQL: https://arxiv.org/abs/2212.06094
- Blogs: Compressed FSM https://www.lmsys.org/blog/2024-02-05-compressed-fsm/ ｜ XGrammar https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar

### 方向 B（RadixAttention + Router）
- vLLM/PagedAttention (SOSP'23): https://arxiv.org/abs/2309.06180
- SGLang: https://arxiv.org/abs/2312.07104
- Preble (ICLR'25): https://arxiv.org/abs/2407.00023 ｜ blog https://mlsys.wuklab.io/posts/preble/ ｜ repo https://github.com/WukLab/preble
- DistServe (OSDI'24): https://arxiv.org/abs/2401.09670
- Mooncake: https://arxiv.org/abs/2407.00079
- CacheBlend: https://arxiv.org/abs/2405.16444 ｜ CacheGen: https://arxiv.org/abs/2310.07240
- S-LoRA: https://arxiv.org/abs/2311.03285
- Blogs: RadixAttention https://www.lmsys.org/blog/2024-01-17-sglang/ ｜ SGLang v0.4 (cache-aware router) https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/

---

*文档生成于 2026-07-20，基于 `dl-main` 分支真实代码路径。阅读中若发现路径/行号漂移（仓库仍在迭代），用 `grep` 重新定位类名即可。*
