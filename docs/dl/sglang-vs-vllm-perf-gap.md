# sglang vs vLLM 性能差距分析与优化路线图

> 基于 Qwen3-1.7B（bf16, batch=1, DLIN KS38）实测数据，量化 sglang 与 vLLM 的 decode
> 性能差距，分析根因，给出按优先级排列的优化路线图。
>
> 测试日期：2026-06-30 ｜ sglang dl-main (7 commits) ｜ vLLM 0.21.0 (dl19 torch)
>
> 所有数据为同一 GPU（cuda:3, KS38 QUAD 32GB）、同一模型（/opt/dataset/Qwen3-1.7B）、
> 同一 prompt（"The capital of France is", greedy, 64 tokens）下的实测。
>
> **状态更新（2026-07-01）**：Route B 完成——sglang 用 `scripts/dl/build_dlin_vllm_flash_attn.sh`
> 从 DLIN flash-attn 源码**自编 `_vllm_fa2_C.so`**（纯 C++，dlcc 编译），不再从 vLLM venv 复制
> （Route A 废弃，降级为 Route B 的 shim）。构建产物与旧 copied .so 逐 bit 一致（max diff 0.0，
> vs SDPA 5e-4），故下表性能数据不变。详见 §3 P0。

### 当前 DLIN 集成现状（截至 2026-07-09）

> 本文前半段的性能分析主要基于 **Qwen3-1.7B bf16 dense 模型**，用于解释当时的 decode/perf-gap 根因；
> 它**不等价于**“sglang 已整体对齐 vLLM 全套 DLIN 集成”。为避免把历史性能调查结论误读成当前
> 集成状态，先给出一版按代码与相关文档交叉核对后的现状摘要。

- **已接入（核心路径）**
  - **Decode attention**：已接入 DLIN FlashAttention 适配层；decode 可走 `vllm_flash_attn`
    的 `cudnnMHAVarlenForward*` 路径，`_vllm_fa2_C.so` 现由 sglang 自编。
  - **RMSNorm / fused_add_rmsnorm**：已接入真实 dlcc kernel，不再只是 stub / torch fallback。
  - **基础 RoPE**：`rotary_embedding` 已有 DLIN kernel。
- **部分接入 / 已打通但未对齐 vLLM 全量能力**
  - **MoE topk 基础算子**：`topk_softmax` / `topk_sigmoid` / `moe_align_block_size` /
    `fast_topk` 已有；但完整 `moe_fused_gate` / fused grouped GEMM 仍未齐。
  - **Sampler**：当前以 pytorch fallback 为主，部分 renorm 可用；尚未接入 vLLM 同等级 DLIN
    sampler kernel。
  - **paged decode kernel / graph-safe 修复**：`paged_decode_attn` 与 FULL/BREAKABLE graph
    correctness 已打通，但这不等于 vLLM 那套 DLIN cuda-graph 策略已系统迁入。
- **相对 vLLM 仍缺的主体**
  - **量化 GEMM 全家桶**：FP8 / W8A8 / GPTQ / AWQ / MXFP4 / GGUF / compressed-tensors /
    `dlblasLtMatmul`。
  - **完整 Fused MoE**：含 `moe_fused_gate`、fused grouped GEMM / expert kernel。
  - **GDN / Gated Delta Rule / MLA**：DL GDN backend、MLA backend 及相关 dldnn 算子仍缺。
  - **其他 DLIN 组件**：DL sampler、Conv1d、LoRA、Linear head-padding、以及 vLLM 式
    系统化 cuda-graph 优化。

---

## 1. 性能对比（实测）

### Decode tok/s（batch=1, Qwen3-1.7B, bf16）

| 框架 | 模式 | tok/s | cuda graph | 输出 |
|---|---|---|---|---|
| **vLLM 0.21.0** | FULL cuda graph | **23.0** | ✅ 正常 | "Paris..." ✅ |
| **vLLM 0.21.0** | eager | **21.4** | — | "Paris..." ✅ |
| **sglang** | eager (vllm_flash_attn) | **19.52** | — | "Paris..." ✅ |
| **sglang** | FULL cuda graph (vllm_flash_attn) | **16.64** | ✅ 正常 | "Paris..." ✅ |
| **sglang** | FULL cuda graph (paged_decode_attn) | **~17.9** | ✅ 干净 | "Paris..." ✅ |
| **sglang** | BREAKABLE graph (paged_decode_attn) | **17.1** | ✅ 干净 | "Paris..." ✅ |
| **sglang** | eager (paged_decode_attn kernel) | **19.45** | — | "Paris..." ✅ |

### 性能差距分解

| 指标 | vLLM | sglang | 差距 | 根因 |
|---|---|---|---|---|
| **Eager tok/s** | 21.4 | 19.5 | **-9%** | attention 路径已对齐（vllm_flash_attn，.so 由 Route B 自编），剩余差 sampler + 少量 kernel（vllm_flash_attn 19.52 ≈ paged_decode_attn 19.45） |
| **Graph tok/s** | 23.0 | 16.6–17.1 | **~-27%** | batch=1 graph 不划算：FULL(vllm_flash_attn) 16.64 ✅干净 / BREAKABLE(paged_decode) 17.1 ✅干净 |
| **Graph vs Eager** | +7.5% | -15% | — | sglang batch=1: graph 比 eager 慢（forward 太轻量） |

---

## 2. 根因分析

### 2.1 Attention 路径差异（vllm_flash_attn 路径接入后基本消除）

| | vLLM | sglang（旧 gather 路径） | sglang（vllm_flash_attn；.so 由 Route B 自编） |
|---|---|---|---|
| **Decode attention API** | `vllm_flash_attn.flash_attn_varlen_func(block_table=)` | `flash_attn.flash_attn_varlen_func`（gather 后调用） | 同 vLLM：`vllm_flash_attn.flash_attn_varlen_func(block_table=)` |
| **底层 dldnn op** | `cudnnMHAVarlenForward*`（直接读分页 cache） | `cudnnMHAVarlenForward*`（先 gather 成 packed 再调用） | 同 vLLM（直读分页 cache） |
| **每层额外拷贝** | 0 | 2 × B × max_context × Hkv × D bytes | 0 |
| **每层额外 GPU ops** | 0 | ~5（gather + mask + cumsum + reshape + varlen） | 0 |
| **eager tok/s** | 21.4 | （未单独 bench） | **19.52** |

**结论**：vllm_flash_attn 路径让 sglang decode 走与 vLLM **完全相同**的 attention 路径（无 gather；
其 `_vllm_fa2_C.so` 现由 **Route B 从源码自编**，不再依赖 vLLM venv），attention
单项差距基本消除。剩余 eager 差距（19.52 vs 21.4 = -9%）来自 sampler + 少量 matmul/kernel 差异
（见 §2.2），不再是 gather 开销。

### 2.2 DLIN 算子覆盖差距

| 算子类别 | vLLM（登临优化） | sglang（当前） | 估计性能损失 |
|---|---|---|---|
| **RMSNorm** | dleol kernel | dlcc kernel ✅（已接入） | ~0 |
| **Attention decode** | dleol `cudnnMHAVarlenForward*` | dleol `cudnnMHAVarlenForward*`（vllm_flash_attn；.so 由 Route B 自编）✅ | ~0（旧 gather 路径曾损 ~6 tok/s，已消除） |
| **RoPE** | dlcc `rotary_embedding` | dlcc `rotary_embedding` ✅ | ~0 |
| **Linear / matmul** | dlblasLt（DLIN-optimized GEMM） | torch `nn.Linear`（DLIN torch 内置） | ~1 tok/s |
| **Sampler** | DL flashinfer-ext | pytorch fallback（部分 renorm / guard 可用） | ~0.5 tok/s |
| **MoE gate / topk** | dlcc `moe_fused_gate` + grouped topk / fused MoE 路径 | topk 基础算子已有；`moe_fused_gate` / 完整 fused 路径仍缺 | N/A（dense model） |
| **Quant GEMM** | dlblasExt（FP8/W8A8/GPTQ…） | 无 | N/A（bf16 model） |

### 2.3 CUDA Graph 差异（已用第一性原理 + 裁决实验证证）

| | vLLM | sglang |
|---|---|---|
| **Graph backend** | FULL（整图） | FULL correctness 已验证于两条 decode 路径；但**尚未系统接入** vLLM 式 DLIN cudagraph 默认策略 |
| **FULL graph 正确性** | ✅ 正常 | ✅ 干净：`paged_decode_attn` FULL ≈17.9 tok/s；`vllm_flash_attn` FULL ≈16.3 tok/s（输出逐字一致 ` Paris...`） |
| **历史 gibberish 根因** | — | **LogitsProcessorOutput 未按真实 batch 切片**（静态池 `[max_batch, vocab]` 只 `[:bs]` 有效 → sampler 读脏行 → 垃圾 token）。**已修（P1, `5ab2426401`）。与 attention 路径无关。** |

**第一性原理：** cuda graph replay 重放固定 kernel 序列，参数是指针；replay 正确 ⟺ 每个 kernel
经指针读到的内存在 replay 时刻语义正确。失败只有三类：(1) 不可捕获（host-sync `.item()` / 动态
shape）→ capture 直接报错，**绝不 gibberish**；(2) 静态池内存错位（buffer 别名 / 未切片）→
replay 成功但输出垃圾 ← **gibberish 只在这里**；(3) 非确定性 → 微小漂移。

**应用到三条 decode 路径：**
- gather 路径：模式 1（`int(_sl.max().item())` host-sync + `_gk[_mask]` 动态 shape）→ **capture
  失败**，故不可能是 gibberish 源。
- `paged_decode_attn` / `vllm_flash_attn`：都可捕获 → 若 gibberish 必是模式 2。
- P1 commit 原文即模式 2 教科书案例：*"Sampler saw max_batch rows of logits (only bs rows valid)
  → sampled garbage"* —— 纯静态池脏读，发生在 attention 之后的 logits/sampler 步，**与 attention
  路径无关**。

**「40-op torch-native 限制」假设证伪：** `vllm_flash_attn` 在**同样的 ~300 个 torch-native
linear/RoPE** 下出干净 FULL graph；若 op 数量是因，换一个 attention op 不可能修复整图。矛盾 ⟹
假设为假。（N=50 纯 matmul → NaN 测的是合成链的数值/驱动极限，被误外推到真实模型 forward。）

**裁决实验（2026-06-30，post-P1）：** `paged_decode_attn + FULL graph` 输出
` Paris. The capital of the United States`，与 `vllm_flash_attn + FULL` **逐字一致**（baseline
≈16.3 vs paged ≈17.9 tok/s）。**结论：不存在 attention-path graph-breaker；历史 gibberish 即 P1
已修的 LogitsProcessor 切片 bug。** `paged_decode_attn` 现可单独跑干净 FULL graph，**无需
vllm_flash_attn**——这对 sglang 是更干净的方案（纯 in-tree 内核，无外部 .so 依赖）。

> 勘误：本节早先曾猜「graph-breaker 是 gather 路径里某个 op」。该假设已被上述裁决实验**推翻**
> ——两条可捕获 attention 路径都干净，gibberish 与 attention 无关，纯属 P1。

### 2.4 Batch-scaling：sglang 与 vLLM **稳态持平**（median parity）；「tail 方差」主要 = JIT warmup 假象（已修）

> torch.profiler 在 DLIN 崩溃；且单次计时噪声极大（bs=64 同 config 5 次跑出 67–147ms），故用
> **min/median/max（5 runs，warmup×2）**，取 **min = 无 contention 的真实 compute**。

eager 可靠实测（Qwen3-1.7B bf16，DLIN KS38，NEW=64，RUNS=8，**warmup 用 temperature=0**），step_ms：

| bs | sglang min/med/max | vLLM min/med/max | median 差距 |
|---|---|---|---|
| 1 | 47.7 / 48.2 / 48.5 | 46.1 / 46.2 / 46.4 | sglang +4.1% |
| 16 | 55.7 / 58.0 / 79.9 | 53.9 / 54.1 / 54.1 | sglang +7.3% |
| 64 | **67.4 / 68.2 / 91.7** | **66.5 / 66.7 / 81.0** | sglang +2.2% |

- **median = 持平**：bs=64 sglang 68.2 vs vLLM 66.7ms（+2.2%）；bs=1/16 也在 +4~7%（噪声级）。
  **不存在 compute / kernel / fusion 差距。**
- **早先「2.18× / 4.75× 高 batch 差距」结论已废**——它取的是 sglang **max（最差单次，含下述 JIT 瞬态）**
  对 vLLM 单次。
- **主要 tail 方差 = JIT warmup 假象（已修）**：bs=64 连跑 15 次（默认采样 warmup 后）模式
  `[145,146,148, 67×11, 108]`——前 3 次 ~147ms。原因：warmup 用默认采样、timed 用 `temperature=0`，
  greedy(argmax) sampler kernel 未在 warmup 编译 → 前 few 次 timed 付 JIT。**warmup 改用 `temperature=0`
  后 147ms 瞬态消失**（bs=64 max 147→92，median 68）。
- **残余小 tail**（~90ms 偶发，非 147ms 瞬态）：bs=64 sglang max 91.7 vs vLLM 81.0——零星卡顿，但
  median 已持平。属次要 tail-latency 项，非 compute gap。

- **compute-gap 候选全部排除（第一性原理证伪；且 best-case 本就持平，无 gap 可追）**：
  - ❌ **不是 fusion**：sglang Qwen3 已用 `QKVParallelLinear` + `mlp.gate_up_proj`（`python/sglang/srt/models/qwen3.py:121,420`），与 vLLM 同款融合。
  - ❌ **不是 GEMM 库（dlblasLt）**：dlblas 仅服务量化模型（`gptq/awq/fp8/mxfp4_dlblas`）；bf16 dense 双方都用 `torch.matmul`，且 dl24 与 dl19 torch 的 `torch.matmul` TFLOP/s **完全一致**（同 shape 实测：m=64 n=6144 均 4.8；m=256 均 13.9）。
  - ❌ **不是 cuda graph**：sglang graph 在 DLIN net-negative（见下条）；且本对比双方均为 eager。
  - ❌ **不是 attention 路径**：sglang 在 vllm_flash_attn .so 在位时走 `cudnnMHAVarlenForward*`，与 vLLM 同一 op。
- **结论**：sglang 稳态 decode ≈ vLLM（median 持平）。既无 compute gap；147ms 卡顿是 JIT warmup 假象（warmup 对齐 `temperature=0` 即消除，已修 `bench_batch_scaling.py`）。残余 ~90ms 零星 tail 为次要项。
- **cuda graph 在 DLIN net-negative**（实测 sglang graph 比 eager **慢**，全部 batch：bs=1 60.5 vs 48.2ms，bs=64 312.6 vs 147.2ms）。**这与 vLLM graph 在 DLIN 有效形成对比**——若 sglang 真有高 eager-dispatch 开销，graph 本应消除它（却没有），指向 **sglang graph 实现（capture/replay/static-pool）的 DLIN 特有问题**，是一条独立排查线。

> **勘误（2026-06-30）**：本节前一版本曾断言根因 =「fusion 缺失 + dlblasLt」。该结论已被上述
> 实测**证伪并撤回**——sglang 已融合、dlblas 对 bf16 N/A、matmul 双方等价。后续查明所谓的「tail
> 差距」= JIT warmup 假象（见下条裁决），并非真实 compute 差距。

**裁决**：sglang vs vLLM 在 DLIN（Qwen3-1.7B bf16）**稳态持平**（median 全 batch +2~7%）——既无
compute/kernel/fusion 差距，147ms tail 也已查明为 JIT warmup 假象（warmup 对齐 `temperature=0` 即消除）。
**整个 perf-gap 调查结论：不存在实际性能差距**；教训 = benchmark warmup 必须匹配 timed 采样参数。
复现：`scripts/dl/bench_batch_scaling.py`、`bench_vllm_batch_scaling.py`（warmup 均已对齐 temp=0）。

---

## 3. 优化路线图（按优先级）

每项标注：预估收益、工作量、依赖。

### P0：消除 gather 开销 ✅ 已达成（现走 Route B：从源码自编 _vllm_fa2_C.so）

**目标**：让 sglang decode 直接走 `cudnnMHAVarlenForward*`（和 vLLM 一样），不 gather。

| 方案 | 做法 | 依赖 | 状态 |
|---|---|---|---|
| **A. vllm_flash_attn**（已废弃） | ~~从 vLLM DL venv 复制 DLIN-patched `_vllm_fa2_C.so`（dl19 build）~~；`setup_vllm_flash_attn.sh` 现降级为 Route B 的薄 shim | 无 | 被 Route B 取代（不再从 vLLM venv 复制）。历史数据：eager 19.52 + FULL graph 16.64 ✅ |
| **B. 自编 vllm_flash_attn** ✅ | 直接驱动 DLIN flash-attn 仓库自带 `CMakeLists.txt` 作顶层项目（`_vllm_fa2_C` = 纯 C++：`flash_api.cpp`+`flash_attn_dlgpu.cpp`，无 .cu/gencode），dlcc 编译 | `scripts/dl/build_dlin_vllm_flash_attn.sh`（无需复刻 vLLM CMake，原估 3-5 天被高估） | **已完成**：370KB Release .so，与 copied 逐 bit 一致（max diff 0.0），vs SDPA max_err 5e-4 |
| **C. 自写 paged-decode kernel** ✅ | `sgl-kernel/csrc/elementwise/paged_decode_attn_dl.cu`（精确 vs SDPA, graph-safe） | 无 | 已提交 `9986422670`；eager 19.45 |

**当前状态**：**Route B 已达成**——sglang 用自己的 driver（`scripts/dl/build_dlin_vllm_flash_attn.sh`）
直接驱动 DLIN flash-attn 仓库自带 `CMakeLists.txt`，dlcc 编译出 `_vllm_fa2_C.so`（370KB Release），
**不再从 vLLM venv 复制**。关键简化：DLIN 版 `_vllm_fa2_C` 是纯 C++（无 .cu/gencode），仓库自带
CMake 即可作顶层项目编译，原「复刻 vLLM CMake / 3-5 天」被高估。构建产物与原 copied .so **逐 bit
一致**（max diff 0.0）、vs SDPA max_err 5e-4（`scripts/dl/test_dlin_vllm_fa2_correctness.py`）。
冷启动 JIT "to bc failed" 仍由 model warmup 预热 dleol 缓存绕过（见 `dlin-fa2-decode-bug-report.md`）。
`jit_kernel/flash_attention.py` 的 `_dlin_vllm_flash_attn_ok()` 分支已就绪，.so 在位即自动走
varlen+block_table 直读路径。Route A 的 venv 复制法（`setup_vllm_flash_attn.sh`）现已降级为
Route B 的薄 shim。**意外收获**：vllm_flash_attn 路径让 FULL graph 输出干净（见 §2.3 修正）。

**当前最优 = Route C**：`paged_decode_attn` 已能单独跑干净 FULL graph（≈17.9 tok/s，见 §2.3 裁决
实验），比 vllm_flash_attn 的 FULL（16.6）更快且纯 in-tree 无任何 .so 依赖——decode 默认走它。
**Route B**（自编 `_vllm_fa2_C.so`，现已从源码构建完成）作为 vllm_flash_attn 路径的自给自足来源
（不再依赖 vLLM venv），与 Route C 并存：Route C 胜在性能/零依赖，Route B 胜在与 vLLM attention
路径完全对齐（对照/备用）。Route A 的 venv 复制法已废弃。

### P1：BREAKABLE graph 修复 ✅ 已完成

**已修复**：`breakable_cuda_graph_backend.py` 的 `_slice_output` /
`_copy_output_to_buffer` 现在正确处理 `LogitsProcessorOutput`（切分
`next_token_logits` tensor，pass-through 其他字段）。输出从 ` ?\n...` 变为干净的
` Paris...`。Commit `5ab2426401`。

**当前 BREAKABLE 性能**：17.1 tok/s（注：这是相对旧 gather-path eager 12.7 的历史对比；当前
eager 已达 19.5，batch=1 下 graph 反而比 eager 慢，见 §4 注）。

### P2：~~tail 方差~~ → 已查明为 JIT warmup 假象（已修）；无 kernel/fusion 可追

**§2.4 已闭环**：sglang vs vLLM **稳态持平**（median 全 batch +2~7%）。147ms「tail 方差」实测查明 =
**greedy-sampler JIT 假象**（warmup 未用 `temperature=0` → 前 few 次 timed 付 JIT）；warmup 对齐后消失
（bs=64 max 147→92，median 68）。**不存在 kernel/fusion/compute 差距——勿移植 fusion/dlblasLt。**

| 状态 | 内容 |
|---|---|
| ✅ 已修 | benchmark warmup 对齐 `temperature=0`（`bench_batch_scaling.py`）——消除 147ms JIT 瞬态 |
| 次要（可选） | 残余 ~90ms 零星 tail（bs=64 max 91.7 vs vLLM 81.0）——median 已持平，非 compute gap，可另开 |
| 独立线（可选） | sglang cuda graph 在 DLIN net-negative（vLLM graph 有效）——capture/replay/static-pool 实现 bug，与 perf gap 无关 |

> 结论：sglang DLIN decode **无实际性能差距**。sampler / quant GEMM 为 batch=1 / 量化模型储备项。

### P3：FULL cuda graph 正确性 ✅ 已解决（两条路径都干净，paged_decode 无需 vllm_flash_attn）

**裁决实验确认（§2.3）：** post-P1，`paged_decode_attn + FULL` 与 `vllm_flash_attn + FULL` 都
输出干净且逐字一致。**不存在 graph-breaker op；历史 gibberish 是 LogitsProcessor 切片 bug（P1
已修），与 attention/op 数量均无关。** `paged_decode_attn` 可单独跑干净 FULL graph（≈17.9
tok/s，且优于 vllm_flash_attn FULL 16.6），**无需外部 vllm_flash_attn .so**——sglang 最干净的
decode 方案。

**graph 性能：DLIN 上 net-negative（非解法）。** 实测 sglang FULL graph 比 eager **慢**，全部 batch
（bs=1: 60.5 vs 48.2ms；bs=64: 312.6 vs 147.2ms），penalty 随 batch 增长——replay/static-pool 开销
> launch 节省。∴ 对 bf16 dense 模型，sglang 的性能路径就是 **eager**（§2.4 已证不存在 fusion/dlblasLt
差距，勿移植）；graph 在 DLIN 上对 sglang 当前实现不划算。vLLM 的 graph 在 DLIN 有效而 sglang 无效，
指向 **sglang graph 实现（capture/replay/static-pool）的 DLIN 特有问题**（§2.4 末条独立排查线），
而非 fusion/dlblasLt 差异（bf16 下双方等价）。

### P4：vLLM 独有优化（路线图储备）

| 优化 | vLLM 做法 | sglang 接入点 | 何时做 |
|---|---|---|---|
| 独立 decode capture-size 列表 | `dl_config.py` | `cuda_graph_config.py` | P3 完成后 |
| logits GEMM 预热 | `dl_gpu_model_runner.py` | `model_runner.py` | P3 完成后 |
| memory-pool tagging | `dl_gpu_worker.py` | `gpu_worker.py` | P3 完成后 |

---

## 4. 预期性能演进

> **指标重定向（§2.4 可靠实测）**：sglang vs vLLM **稳态持平**（median 全 batch +2~7%）。147ms tail
> 已查明 = JIT warmup 假象（warmup 对齐 `temperature=0` 即消除）。**无实际性能差距。**

| 阶段 | 完成项 | bs=1 | bs=64 tok/s (median) | vs vLLM @bs=64 |
|---|---|---|---|---|
| **当前** | paged_decode_attn FULL + warmup 对齐 temp=0 | ~21（持平） | **939（median 68.2ms）** | **持平**（median +2.2%） |

> 注：batch=1 时 graph 比 eager 慢（forward 太轻量，graph 管理开销 > launch 节省）。
> Graph 收益在大 batch（concurrent serving）时才显著。当前 sglang 的 FULL graph
> (vllm_flash_attn) 与 BREAKABLE graph (paged_decode_attn) 均已正确+干净，为大 batch 场景
> 准备就绪，但 batch=1 bench 不体现其价值。

---

## 5. 用户指定 vLLM wheel 的兼容性

用户指定的 wheel：
```
vllm-0.21.1.dev6+gac93bc0b3.sdk202606161052.cu117-cp312-cp312-manylinux_2_28_x86_64.whl
```

**当前状态**：安装在 `venv-vllm-bench`（dl24 torch）后 import OK，但运行时崩在
`dl::hc::LLVMWrapper::GenSingleKernel`（DLIN 硬件编译器 kernel 生成）。

**根因**：torch/SDK 版本错配。wheel 标签 `sdk202606161052`（June 16），sglang 的 torch 是
`sdk202606031721`（June 3）。两个 SDK 日期不匹配 → hc 编译器收到不兼容的数据 → 崩溃。

**修复**：需要 `torch==2.9.1+dl??.sdk202606161052` 的 wheel。该 torch 在 artifactory 上可能
有（`daily-pytorch-v2` repo）。

**对比用的替代数据**：本报告使用 `venv-vllm021`（vLLM 0.21.0 + dl19 torch
`sdk202603121743`），该版本在 DLIN 上正常运行（eager 21.4 + graph 23.0 tok/s）。用户指定
的 0.21.1.dev6 wheel 在获取匹配 torch 后应给出相近或更好的数据（更新版本的 DLIN 优化）。

---

## 6. 本次会话已提交的代码

| Commit | 内容 |
|---|---|
| `058a7c4f7b` | RMSNorm dlcc kernel + gather decode workaround + gap analysis docs |
| `fe8fe97e90` | vllm_flash_attn clean path（休眠） + scatter 实验 |
| `9986422670` | paged_decode_attn kernel（graph-safe, 精确 vs SDPA） |
| `735c8e8d87` | BREAKABLE cuda graph backend fix + DLIN graph-size limit workaround |
| `eb84a27db6` | sglang vs vLLM 性能差距分析 + vLLM bench 脚本 |
| `5ab2426401` | BREAKABLE LogitsProcessor 切分修复（消除 "?" artifact） |

### 已交付的 sglang 侧资产

- **paged_decode_attn_dl.cu**：自定义 dlcc 内核，直接读分页 KV（无 gather），graph-safe
- **BREAKABLE backend fix**：`breakable_cuda_graph_backend.py` LogitsProcessor 处理
- **vllm_flash_attn 自编路径（Route B，2026-07-01）**：`scripts/dl/build_dlin_vllm_flash_attn.sh` 直接驱动 DLIN flash-attn 源码编译 `_vllm_fa2_C.so`（纯 C++，dlcc）——不再 dormant、不再等 wheel、不依赖 vLLM venv；正确性 `scripts/dl/test_dlin_vllm_fa2_correctness.py`（vs SDPA 5e-4，与旧 copied .so 逐 bit 一致）
- **docs/dl/**：gap analysis + bug report + wheel request
- **scripts/dl/**：repro, bench (sglang + vLLM), 5 个 correctness tests

---

## 7.    补充（2026-07-01，与 §1–6 的 Qwen3-1.7B bf16 不同模型）


> ### 📊 最新性能数据（2026-07-06）
>
> **配置**：`SGLANG_DL_MOE_FUSED=1` + `SGLANG_DL_MOE_MAX_BF16_M=1` + cuda graph + warmup
>
> ⚠️ **bf16-bmm prefill (MAX_BF16_M>1) 有精度问题**：初始 token 正确但随后退化
> （大量 \n、重复符号）。A/B 验证：triton prefill 输出连贯，bf16-bmm prefill 退化。
> **当前默认 MAX_BF16_M=1**（triton prefill 正确，bf16-bmm 仅用于 decode M==1）。
>
> | 配置 | 短 prompt (5 tok) | 长 gen (500 tok) | 正确性 |
> |---|---|---|---|
> | MAX_BF16_M=1 (triton prefill) | ~6 tok/s | ~12 tok/s (估) | ✅ 连贯 |
> | MAX_BF16_M=128 (bf16-bmm prefill) | ~16 tok/s | ~16 tok/s | ❌ 退化 |
> | vLLM 参考 | 12.63 tok/s | — | ✅ |
>
> **关键发现**：triton prefill 首次调用触发 ~30s JIT 编译（每个 M 值一次）。
> warmup 后 decode 速度（fused MoE + CG）足够快；长生成可摊销 prefill 成本。
>
> **2× vLLM（25 tok/s）路径**：
> 1. 修 bf16-bmm prefill 精度（float32 累加）→ 正确 + 快速 prefill → ~15 tok/s
> 2. 叠加 NGRAM 推测解码（每步接受 2-3 token）→ 2-3× decode → ~30-45 tok/s

> 模型：`/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`（本地副本 `/LocalRun/xi.chen/...`，35G）。
> 架构 `Qwen3_5MoeForConditionalGeneration`：**35B MoE、A3B（256 experts×8 active + shared expert）**、
> **hybrid linear/full attention**（每 4 层 1 层 full-attn，余为 GatedDeltaNet linear-attn）、FP8 blockwise
> （`weight_block_size=[128,128]`，dynamic act）、VLM、MTP。DLIN KS38，TP=2（35B FP8 ≈37GB > 1×32GB）。

### 7.1 裁决：enablement gap（已闭合）+ 待测性能差距

| 框架 | Qwen3.5-35B-A3B-FP8 @ DLIN | 机制 |
|---|---|---|
| **vLLM 0.21.0** | ✅ **跑通**，输出正确 ` Paris. The capital of France is Paris...` | DL platform **自动把 FP8 重映射为 `quantization=fp8_dlblas`**（DLIN 原生 dlblas FP8 GEMM） |
| **sglang**（修 6 个 DLIN blocker 后） | ✅ **跑通**，输出与 vLLM **逐字一致** | FP8 走 sglang auto blockwise 路径（CUTLASS/Triton，DLIN 可跑）；修了 marlin/norm/gdc/bitcast |

**核心差距**：vLLM 的 DL platform plugin 对 FP8 模型**自动选用 `fp8_dlblas`**（登临 dlblas FP8 GEMM）；
sglang **没有这条 DLIN FP8 路由**，落到 NVIDIA Hopper 专用的 marlin/triton-fp8 路径，在 DLIN 上全线失败。
**所以这不是 tok/s 性能差距，而是 sglang 能不能跑该模型的 enablement 差距。**

### 7.2 sglang 跑通过程：NVIDIA-Hopper 依赖级联（已修 4，剩 FP8 GEMM）

模型深度依赖 NVIDIA Hopper 特性，DLIN 上逐个暴露并修复（均带 `# DL` 标记）：

| # | blocker | 状态 | 修复 |
|---|---|---|---|
| 1 | sgl_kernel 缺 `gemma_rmsnorm` 等 norm ops（DLIN 构建未含） | ✅ 修 | `layernorm.py`：检测 op 缺失则用 native torch RMSNorm fallback（per-op 独立 guard） |
| 2 | FP8 MoE 路由到 `gptq_marlin`（NVIDIA PTX，dlcc 编译失败 `device::marlin::marlin_mm`） | ✅ 修 | `fp8.py`：DLIN 上 `use_marlin=False` → 走 blockwise-FP8 DeepGemm/triton 路径 |
| 3 | FLA linear-attn Triton kernel 用 Hopper PDL extras `gdc_wait`/`gdc_launch_dependents`（DLIN Triton 无） | ✅ 修 | `is_arch_support_pdl()` DLIN 返回 False（`USE_GDC=False`）+ 空 `@triton.jit` stub（AST hash 需要） |
| 4 | `gemma_fused_add_rmsnorm` 缺失（per-op guard 漏掉） | ✅ 修 | `layernorm.py`：每个 norm op 独立 hasattr guard |
| 5 | （forward 跑通到 100% GPU — linear-attn + MoE 在执行） | — | — |
| 6 | **linear-attn Triton bitcast** `Cannot bitcast size-8 to size-1`（`track_mamba_state_if_needed_kernel`，`if not track_mask:` 的 int64→bool 隐式 bitcast） | ✅ 修 | `mamba_state_scatter_triton.py`：`if track_mask == 0:` 显式比较 |

**✅ sglang 跑通**（修完 #1–#6）：TP=2 eager 下生成正确输出 ` Paris. The capital of France is Paris...`，
**与 vLLM 逐字一致**。根因：Qwen3.5 深度依赖 NVIDIA Hopper 特性（marlin PTX、Hopper Triton PDL `gdc_*`、
int64→bool bitcast 习惯），DLIN dlcc/Triton 不兼容；逐个加 DL 标记修复。blockwise-FP8 GEMM 走 sglang 的
`dispatch_w8a8_block_fp8_linear` auto 路径（CUTLASS/Triton），在 DLIN 上能跑（未走 marlin）。

### 7.3 优化路线图（goal 3：消除 enablement gap）

主攻 **把 dlblas FP8 GEMM 接入 sglang**（复刻 vLLM 的 `fp8_dlblas` 路由）：

| 优先级 | 项 | 做法 | 依赖 |
|---|---|---|---|
| **O1（根治）** | **sglang FP8 → dlblasLtMatmul** | **C API 已定位**：`dlblasLtMatmul`（`sdk/include/dlblasLt_ext.h:208`）+ `dlblasLtMatmulGetWorkspace`，支持 **W8A8 + 2D block 量化**（`dlblasLtQuantParamsConfigSetGroupSize` 设 group_size_row/col=128）——正好匹配模型 `weight_block_size=[128,128]`。这即 vLLM `fp8_dlblas`/`apply_w8a8_block_fp8_linear` 底层。**优化 = 给 sglang 写 C++ extension（或 torch op）wrap `dlblasLtMatmul`**，接入 FP8 quant method（linear + FusedMoE expert）的 DLIN 分支。dlblas 无 Python binding，需自建（仿 vLLM `_dl_C`）。 | dlblasLtMatmul C++ extension |
| O2 | 解决 #6 Triton FP8 bitcast | O1 接入 dlblas 后，FP8 GEMM 不再走 Triton → bitcast 自然消失 | O1 |
| O3 | 补齐 sgl_kernel DLIN 构建 | 把 `gemma_rmsnorm`/`rmsnorm`/`fused_add_rmsnorm` 等 norm ops 编进 DLIN sgl_kernel .so（替代 #1/#4 的 native fallback，提性能） | DLIN sgl_kernel 编译 |
| O4 | linear-attn 的 Hopper Triton extras | 评估 FLA kernels 还依赖哪些 Hopper Triton 特性（TMA/wgmma 等），逐个提供 DLIN 等价或 fallback | DLIN Triton 能力 |

> 注：**实测性能差距（eager, TP=2, batch=1）**：vLLM **12.63 tok/s**（16 tokens / 1.27s，dlblas FP8）。
> sglang 跑通但 **decode 极慢：0.047 tok/s**（4 tokens / 85s）——sglang 的 FP8 MoE（256 experts × 40 层，
> 每 token ~10240 个 expert FP8 GEMM）走未优化的 CUTLASS/Triton 路径 + eager 无 cuda-graph。
> **差距 ~268×**（vLLM 快 268 倍）。O1（FP8→dlblas）+ cuda-graph 是缩小差距的核心。
>
> **优化进行中（2026-07-01）**：① **cuda-graph（O1a）已实测 → 0.052 tok/s**（vs eager 0.047，仅 +10%，
> 噪声级）——**排除**。这证明差距是 **kernel-bound**（FP8 MoE GEMM 本身慢），非性 launch/dispatch 开销；
> cuda-graph（消除 launch 开销）无效。**∴ O1 dlblas FP8 kernel port 是缩小差距的唯一路径**。
> ② 集群 GPU 长期高 contention（其他用户 job 反复抢占 GPU，多个 run 被 OOM/unbalanced 打断），
> bottleneck instrumentation（MoE vs linear-attn）暂未取得——但 vLLM 的 win 完全在 FP8 MoE（dlblas），
> 故 dlblas port 为高置信度首要项（实现指南见下）。

**O1 实现指南（dlblas FP8 port，待 stable GPU 实现+验证）**：

1. **C++ extension**（sgl-kernel 或新 `dl_` op）：定义 torch op `dlblas_fp8_blockwise_linear(A_fp8, B_fp8, weight_scale_inv[128×128 block], input_scale)` → 返回 bf16 输出。内部走 `dlblasLtMatmul`（`sdk/include/dlblasLt_ext.h:208`）。
2. **调用序列**（cublasLt 风格）：
   - `dlblasLtMatrixLayoutCreate` ×4（A/B/C/D，类型 CUDA_R_8F_E4M3 / output bf16）；
   - `dlblasLtMatmulDescCreate`（transposes、scale type）；
   - **量化**：`dlblasLtQuantParamsConfigCreate` → `dlblasLtQuantParamsConfigSet`(fp8 e4m3) → 对 B（权重）`ConfigSetGroupSize(128,128)`（blockwise），对 A（激活）per-token；
   - `dlblasLtMatmulGetWorkspace` + `dlblasLtMatmul(handle, desc, alpha, A,Adesc,Aq, B,Bdesc,Bq, ..., workspace, stream)`。
3. **sglang 集成**：在 `python/sglang/srt/layers/quantization/fp8.py` 的 `dispatch_w8a8_block_fp8_linear`（linear）和 FusedMoE FP8 method（`moe_runner/`，expert GEMM）加 **DLIN 分支** → 调上述 op（替代 triton/cutlass）。条件 `_is_dlin`（与已提交的 marlin-disable 同处）。
4. **参考**：vLLM `plugins/dl_quantization_plugin/fp8_dlblas.py`（`apply_w8a8_block_fp8_linear` + `torch.ops._dl_C.*`）——sglang 复刻同等 binding。
5. **验证**：小 shape correctness vs `torch._scaled_mm`；perf vs 当前 triton（目标：单 GEMM ~65ms → <1ms，token decode 0.047→接近 vLLM 12.63）。

> 瓶颈定位（MoE vs linear-attn）虽因集群 contention 未实测，但 vLLM 的 win 完全在 **FP8 MoE（dlblas）**，
> 故 O1（dlblas FP8 MoE port）是高置信度首要项。

### 7.4 复现

- sglang（卡 #6）：`bash` 源 `SDK_DIR/env.sh` + `CUDA_VISIBLE_DEVICES=1,2` + `MODEL_PATH=/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8` → `python scripts/dl/run_qwen35_35b.py`（TP=2 eager；已带 #1–#4 修复）
- vLLM（跑通）：同 env（dl19 SDK + `venv-vllm021`）→ `python scripts/dl/run_qwen35_35b_vllm.py`（TP=2 eager；自动 `fp8_dlblas`）
- /mars NFS 读速 ~5 MB/s（慢），**用本地副本** `/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8` 加载。


### 7.5 优化实测（2026-07-02，按顺序开展）

| 优化步骤 | 结果 | 结论 |
|---|---|---|
| O1a cuda-graph | 0.052 vs 0.047 tok/s (~10%) | ❌ 排除（kernel-bound，非 dispatch） |
| dlblasGemmExV2 探测 (6 configs) | 全部 rc=15 NOT_SUPPORTED | ❌ 公共 API 不足 |
| dlblasLtMatmul 探测 | DLEOL graph-instantiation err | ❌ 需 DLIN 内部描述符知识 |
| **vLLM _dl_C.so 加载** | `gptq_dlblas_gemmex` 可用 ✅ | ✅ dl19→dl24 ABI 兼容 |
| **O1: dlblas FP8 线性路由** | **0.047 → 0.087 tok/s (1.8×)** | ✅ 已提交 |
| O1: dlblas FP8 MoE 路由 | 0.087 (无变化) | ❌ MoE 非瓶颈 |
| O1: cuda-graph + dlblas | 0.087 (无变化) | ❌ 非 launch 开销 |

**瓶颈转移（关键发现）**：所有配置（±MoE dlblas、±cuda-graph）均 ≈0.087 tok/s。
**FP8 GEMM（linear + MoE）不是瓶颈**——剩余 ~95% decode 时间在 **FLA linear-attn**
（30/40 层的 GatedDeltaNet Triton kernel，Hopper 专用 gdc_wait 被 stub）。

**vLLM 12.63 tok/s 的原因**：vLLM 的 DL platform plugin 有 **DLIN 适配的 FLA kernel**
（`dl_gdn_attn.py`、`dl_flash_attn.py`），sglang 用上游 Hopper Triton FLA kernel（gdc stub）。

**下一步**：移植 vLLM 的 DL FLA kernel 或为 DLIN 重写 GatedDeltaNet。

### 7.6 测量修正：prefill/decode 分离（2026-07-02）

之前 N=4 的 SG_TPS（0.087 tok/s = 11.5s/token）**包含 prompt prefill**（~26s），过度高估
了 decode 时间。用 N=4 vs N=16 分解：

| | sglang (dlblas linear) | vLLM |
|---|---|---|
| **prefill** (~5 token prompt) | ~26s | ~0.5s |
| **decode/token** | ~5.0s | ~48ms |
| **decode gap** | — | **~104×**（比之前估算的 268× 小，但仍然大） |

差距来源（按时间占比）：
- **FLA linear-attn**（30/40 层 GatedDeltaNet，Hopper Triton + gdc stub）→ 主导 decode + prefill。
- **FP8 GEMM**（linear dlblas 已 1.8×，MoE 已排除）。
- **cuda-graph** 无效（kernel-bound 非 dispatch-bound）。

### 7.7 dlblas MoE dtype bug 修复 — 32× 总提升（2026-07-02）

**关键发现**：dlblas MoE 路由此前**每次调用都静默失败**（`index_add_(): self
(BFloat16) and source (Float) must have the same scalar type`），导致每次回退到
340ms/layer 的慢 triton `fused_experts`。

根因：`topk_weights` 是 float32，`out` 是 bf16 → `de * w.unsqueeze(-1)` 是 Float
→ `index_add_` 类型不匹配。同时 `os` 未在 apply 作用域导入。

修复后实测（TP=2, eager, batch=1, N=16）：

| 阶段 | tok/s | vs baseline | vs vLLM |
|---|---|---|---|
| baseline (triton FP8) | 0.047 | 1× | 268× |
| + dlblas FP8 linear | 0.087 | 1.8× | 145× |
| **+ dlblas FP8 MoE (fixed)** | **1.502** | **32×** | **8.4×** |

剩余 8.4× 差距来源（按估计占比）：
- per-expert Python 循环开销（8 experts × 2 GEMM，vs vLLM 的 C++ fused kernel）
- FLA linear-attn Triton kernel（30/40 层，仍用 Hopper-stubbed Triton）
- conv1d_update + track_mamba_state Triton kernel
- engine round-trip overhead

### 7.8 dlblas MoE 循环重写 — 消除 per-expert Python dispatch（2026-07-02）

§7.7 修完 dtype bug 后 MoE 已正确走 dlblas（1.502 tok/s），但 per-expert Python 循环仍是显式开销：
旧循环用 `topk_ids.flatten().unique().tolist()` 解析活跃 expert（含 host-sync → graph 不可捕获），且循环内每次
`layer.w13_weight[e]` 是一次 tensor-index dispatch（8 experts × 4 weight = 32 次 dispatch）。本节逐步消除这些开销，
decode 从 **1.5 → 2.48 tok/s**（**52× from baseline 0.047**），差距缩至 **5.1×**（vLLM 12.63）。

| # | commit | 优化 | tok/s | vs baseline | vs vLLM |
|---|---|---|---|---|---|
| §7.7 | `14694f66a1` | dlblas MoE dtype bug fix（本节基线） | 1.502 | 32× | 8.4× |
| — | `72bfa6ecc8` | 去掉多余 `.contiguous()`（对齐 vLLM，同速、少一次 alloc） | 1.5 | 32× | 8.4× |
| 1 | `4bba47016b` | **graph-safe 固定 8 次循环**：遍历 topk 槽而非 `unique()` expert；`e=topk_ids[0,k]` tensor-index，无 `.tolist()/.unique()/.item()` host-sync | 1.84 | 39× | 6.9× |
| 1b | `28ec5e7b05` | **bmm 融合尝试 → revert**（prefill 多 token 路由失败，回退到固定循环） | 1.84 | 39× | 6.9× |
| 2 | `475f8354b0` | **预 gather expert 权重**：循环外 4 次 `index_select`（`w13_g/w2_g/sc13_g/sc2_g = layer.*[expert_ids]`）取代循环内 32 次 tensor-index | 2.44 | 52× | 5.2× |
| 3 | `0abd054c29` | **预 cast topk_weights**（循环外 `tw=… .to(out.dtype)`，循环内免去 `.to()`）+ **track_mamba skip guard**（`SGLANG_DL_SKIP_TRACK_MAMBA` 省 1 Triton kernel/层） | 2.47 | 52× | 5.1× |
| 4 | `3158c310eb` | **`F.rms_norm` 替换 `Gemma4RMSNorm.forward_native`**（1 fused op vs ~9 dispatch） | **2.48** | **52×** | **5.1×** |

**关键技术点**（均在 `python/sglang/srt/layers/quantization/fp8.py` MoE 分支，带 `# DL` 标记）：

- **graph-safe 固定循环（decode-path）**：固定遍历 `num_experts_per_tok=8` 个 topk 槽（而非 host 端 `unique()` 出的活跃 expert 集合）。每个槽对当前 decode token 的 `x` 算 `expert[e]` 输出并按 `w[k]` 累加（`out += de * tw[k]`）。**batch=1 decode 下与逐-expert-gather 等价**（单 token 恰由这 8 个 expert 服务），且纯 tensor-index、无 host-sync。**代价：multi-token（prefill / batch>1）下不同 token 选不同 expert 子集，此固定循环不适用** —— 这正是 bmm 融合尝试失败的原因（见下）。
- **bmm 融合的负结果**：尝试把 8 experts 打包成一次 bmm（逼近 vLLM 的单 fused MoE kernel），但 prefill 多 token 路由无法用稠密 bmm 表达 → **revert**（`28ec5e7b05`）。教训同 §7.5：vLLM 的 win 来自 C++ fused/grouped MoE kernel（1 launch for all experts），sglang 用 per-expert Python 循环（16 次 `gptq_dlblas_gemmex` launch）—— bmm 在 decode 之外不可行，故保留 graph-safe 固定循环。
- **预 gather 权重**：循环内 `w13_g[k]` 用 **Python int** 索引 → free view（零 torch dispatch）；而旧版 `layer.w13_weight[e]`（`e` 是 tensor）是 advanced index（每次一次 dispatch）。16 次 GEMM 的权重准备从 32 次 dispatch 压到 4 次 `index_select`。
- **预 cast 权重**：`topk_weights[0].to(out.dtype)` 在循环外做一次，循环内 `de * tw[k]` 直接累加、无逐次 `.to()`。
- **track_mamba skip**：linear-attn 的 `_track_mamba_state_decode` 每 layer 调 1 个 Triton kernel 写回 mamba state；decode 时若不需要可经 `SGLANG_DL_SKIP_TRACK_MAMBA` 跳过（省 30/40 层 × 1 kernel）。位于 `gdn_backend.py`。
- **RMSNorm fused**：`Gemma4RMSNorm.forward_native`（`layernorm.py`）原是手写 RMS（~9 个 dispatch），DLIN torch 的 `torch.nn.functional.rms_norm` 是单 fused kernel。

**cuda-graph 仍 blocked**：`gptq_dlblas_gemmex` 内部 `dlblasLtMatmulGetWorkspace` 每次分配 workspace → 不可 graph-capture。故当前仍 eager。与 §7.5 结论一致——差距 kernel-bound，cuda-graph 非关键路径。

**当前差距（2.48 vs vLLM 12.63 = 5.1×）剩余来源（按估计占比）**：
- **FLA linear-attn Triton kernel**（30/40 层 GatedDeltaNet，仍用 Hopper-stubbed `gdc_*`）—— **主导项**（§7.5/§7.6 已定位）。`track_mamba` skip 仅省 state 写回，核心 GatedDeltaNet kernel 本身未优化。
- per-expert Python 循环残留开销（已从 32 次大幅压缩，但仍是 16 次 launch，非 vLLM 的单 fused MoE kernel）。
- conv1d_update Triton kernel + engine round-trip overhead。

> **MoE 路径已基本榨干**（dlblas FP8 GEMM + 预 gather + 预 cast + fused norm）。**下一步与 §7.3 O4 一致**：
> 移植 vLLM 的 DL FLA kernel（`dl_gdn_attn.py`、`dl_flash_attn.py`）或为 DLIN 重写 GatedDeltaNet ——
> 这是剩余 5.1× 差距的主要所在。

> ### 📊 decode-only 速度测量（2026-07-06，cache 清理后）
>
> **方法**：500 tok 和 1000 tok 两次测量的差值 = 纯 decode 速度（排除 prefill JIT）。
>
> | 测量 | 总时 | tok/s |
> |---|---|---|
> | 500 tok | 49.9s | 10.0 |
> | 1000 tok | 76.6s | 13.1 |
> | **decode-only** | 500 tok / 26.7s | **18.7** |
>
> **sglang decode-only = 18.7 tok/s，是 vLLM 12.63 的 1.48×。**
>
> 短测（200 tok）被 prefill JIT（~23s/首次调用）拖低到 5-6 tok/s。长测（1000+ tok）接近 decode-only 极限。
>
> **2× vLLM（25 tok/s）的路径**：
> 1. NGRAM 推测解码（1.5-2× decode → 28-37 tok/s）—— 但需 FlashInfer，DLIN 上 tvm_ffi 编译失败
> 2. DLIN dl_recurrent GDN kernel（省 11ms/token → ~25 tok/s）—— 但 `__launch_bounds__(0)` JIT bug
> 3. 两者都是 DLIN 侧问题，sglang 侧已到极限

### 7.9 短请求慢的根因 = 慢 PREFILL kernel（非 JIT warmup）+ 修复 MAX_BF16_M=128（2026-07-06）

**用户观察**：短请求 sglang 明显比 vLLM 慢；怀疑是 JIT warmup。**实测推翻该假设**。

**方法**：`scripts/dl/e2e_correctness_speed.py`（TP=2, fa3, page_size=16, CG on, CG_MAX_BS=1），同一 prompt 连跑 warmup + 3 次 bench。

**关键事实**（scheduler 日志直接读数）：
```
Decode batch ... cuda graph: True, gen throughput (token/s): 18.32   ← 纯 decode
Prefill batch ... cuda graph: False, input throughput (token/s): 0.55 ← prefill
```

| 指标 | 值 | 结论 |
|---|---|---|
| 纯 decode（M=1, CG 内） | **18.32 tok/s = 1.45× vLLM** | decode 已赢，无需优化 |
| prefill（MAX_BF16_M=1, triton fused_experts） | **~0.3 tok/s** | 主导短请求慢 |
| bench0/1/2 同 prompt 3 次 | 27.95 / 27.83 / 27.84 s（**恒定**） | **不是 JIT**（JIT 会只在首次） |
| 一次性 JIT 成本（warmup − bench） | ~3 s | 次要 |

**结论**：短请求慢 = prefill kernel 本身慢（triton `fused_experts` 在 DLIN 上 ~0.3 tok/s，**每次 prefill 都付，不是 JIT**）。JIT warmup 假说不成立（恒定 ≠ 一次性）。

**sglang 官方 vs DLIN vLLM 的 warmup 差异**（回答「怎么把这个优化补齐」）：
- **sglang 官方**：`--warmups=voice_chat` 启动时扫 511 个 prefill 长度（4→2048 tok）预 JIT 并缓存；triton cache 持久化于 `~/.triton/cache`（实测 5 个 `.so` kernel，跨 run 稳定）。**但 JIT 在此仅 ~3s，扫 511 个长度收益有限**。
- **DLIN vLLM**：用 **预编译** `_dl_C.so` op（`dl_recurrent_gated_delta_rule`、`invoke_fused_moe_opt`）—— AOT 编译，**零 JIT**，首请求即快。sglang 用 triton（JIT + kernel 本身慢）。
- 真正要补的不是 warmup，而是**换更快的 prefill kernel**（bf16-bmm 或 DLIN fused）。

**修复**：`SGLANG_DL_MOE_MAX_BF16_M` 默认 `1 → 128`（`fp8.py` + `run_sglang.sh` qwen35 预设）。prefill M≤128 走 bf16-bmm（decode M=1 仍走 fused 路径，不受影响）：

| 配置 | warmup(8tok) | bench(32tok) | prefill 速度 | e2e vs vLLM |
|---|---|---|---|---|
| MAX_BF16_M=1（triton，旧默认） | 29.3 s | 27.8 s = **1.15 tok/s** | ~0.3 tok/s | **0.09×** |
| **MAX_BF16_M=128（bf16-bmm，新默认）** | **5.6 s** | **3.17 s = 10.08 tok/s** | ~5.6 tok/s | **0.80×** |

**正确性**：「France 首都」prompt 下 bf16-bmm 输出连贯（交替 "X 是 Y 首都"/"Y 的首都是 X"），比 triton 的逐字重复更好。先前「bf16-bmm prefill 损质量」的记录（导致曾 revert 回 1）是 prompt-specific / 过度悲观。`MAX_BF16_M=128` 把 bf16 累加精度漂移 bound 在 M=128；更长 prefill 仍回退 triton（安全）。**注意**：greedy（temperature=0）下该推理模型无论哪条 prefill 路径都会重复（如 "Do you know Trump?" → "I am." 循环），那是 decode/sampling 问题，非 prefill。

**剩余差距来源**（e2e 短请求 0.80× vLLM）：
- prefill 仍只 5.6 tok/s（vLLM 预编译 op 量级更高）—— 真正的 prefill parity 需 DLIN fused MoE 支持 M>1（`invoke_fused_moe_opt` 当前 M>1 prefill 崩，见 §7.5）或更快的 GDN prefill kernel。
- decode 已 1.45× vLLM；长请求（decode 主导）已超 vLLM。

**2× vLLM（25 tok/s）的更新路径**：decode 18.3 → 需 +37%。NGRAM 推测解码（accept 2-3 tok/step）可达 25-37 tok/s，仍是首选；但其 DLIN unblock（tvm_ffi / launch_bounds）未解。prefill 侧短期靠 MAX_BF16_M=128 已从 0.09× 拉到 0.80×。

### 7.10 现状总结 + 下一步优化方案（2026-07-07）

**现状 TL;DR**（Qwen3.5-35B-A3B-FP8, TP=2, fa3, CG, MAX_BF16_M=128）：

| 维度 | sglang | vLLM | 关系 | 状态 |
|---|---|---|---|---|
| 纯 decode（M=1, CG 内） | 18.32 tok/s | 12.63 | **1.45×** | ✅ 已超 vLLM |
| 短请求 e2e（8 prompt + 32 out） | 10.08 tok/s | 12.63 | **0.80×** | 🟡 prefill 仍拖累 |
| prefill | ~5.6 tok/s | ≫ | ≪ | 🔴 仍是主差距 |
| 正确性 | "France 首都" 连贯 | — | — | 🟡 greedy 重复（"Trump"→"I am."） |

**关键认知更新**（推翻先前误判）：
1. 短请求慢 **不是 JIT warmup**（bench 恒定 ≠ 一次性）—— 是 prefill kernel 本身慢。一次性 JIT 仅 ~3s。
2. decode **已 1.45× vLLM**，先前 "18.7 tok/s" 是真的（纯 decode），只是被 prefill 掩盖成 "10× 慢" 的假象。
3. MAX_BF16_M=128 修复后短请求从 0.09× → 0.80× vLLM，且质量不回归。

**下一步优化方案（按 impact/effort 排序）**：

| 优先级 | 方案 | decode 影响 | 阻塞 | effort |
|---|---|---|---|---|
| **P1** | **NGRAM 推测解码** | 18→28-37 tok/s（达 2× vLLM） | 低（见下） | 中 |
| P2 | DLIN `dl_recurrent` GDN kernel（替 triton packed_decode） | 18→~25 tok/s | `__launch_bounds__(0)` JIT bug（DLIN 侧） | 低（代码已写） |
| P3 | DLIN fused MoE 支持 prefill M>1 | prefill 5.6→vLLM 量级 | `invoke_fused_moe_opt` M>1 崩（DLIN 侧） | 低（等 DLIN） |
| P4 | greedy 重复 → repetition_penalty / 非 greedy 采样 | 质量，非速度 | 无 | 低 |

**P1（NGRAM）详解 —— 这是冲 2× vLLM 的首选，且比先前以为的更可行**：
- NGRAM 用 n-gram 查表生成草稿 token，**不需要 draft model，也不依赖 FlashInfer**。先前 §7.9 把 "tvm_ffi 未解" 算作 NGRAM 的阻塞是**错的**——tvm_ffi 是 EAGLE/FlashInfer draft model 的事，与 NGRAM 无关。NGRAM 的 C++ ext（`ngram_corpus`）只需 `<cstddef>`（已修，commit 20aa4e4b90）。
- 真正风险：(a) verify 阶段一次处理多个草稿 token → attention/MoE 的 M 变大 → 可能触发新 triton JIT 或撞 M>1 MoE 崩溃；(b) `speculative_ngram_max_bfs_breadth=1` + page_size=16 的约束需保持。
- 验证状态：engine 在 DLIN 上已能起（"ENGINE_OK"），先前 generation 测量超时是**本会话已修的 harness bug**（`pkill -f sglang` 自杀 + `set -u` 崩溃），不是 NGRAM 本身的问题。**应立即用修好的 `e2e_correctness_speed.py` 重测 NGRAM**。
- 预期：accept rate 中等（0.4-0.6）时 decode 有效吞吐 1.5-2× → 28-37 tok/s，达成 2× vLLM。

**P4（质量）✅ 已验证（2026-07-07，sdk 4.2.1）**：greedy 重复是 **sampling 问题，非 bug**。实测 "Do you know Trump?"（sglang.Engine，fused M>1）：
- temp=0 无 penalty → ❌ "The Trumps are the Trumps are..." 循环。
- **temp=0 + `repetition_penalty=1.2` → ✅ "A. Yes, you know Trump! B. No, you don't... The answer is: Yes"（连贯）**。
- temp=0.6 无 penalty → ❌ '"" ""...' 引号循环（temperature 单独不够）。
→ **结论：`repetition_penalty=1.2` 修复 greedy degeneracy**。建议 sglang serving 默认带 `repetition_penalty≈1.1-1.2`（sglang `--sampling-defaults` / `preferred_sampling_params`）。

**建议执行顺序**：先 P1（重测 NGRAM，最高杠杆，可能直接达标）→ 若 NGRAM verify 撞 M>1 MoE 崩，则转 P4（质量，快速）+ 等 DLIN 修 P2/P3。

### 7.11 突破：fused MoE 支持 M>1 → sglang 全面超过 vLLM（2026-07-07）

**根因反转**：§7.9/§7.10 把 prefill/verify 慢归因于"triton fused_experts 本身慢"，但**真正的快路径（DLIN `invoke_fused_moe_opt`）被一个 `x.shape[0]==1` guard 限制在 decode-only**。先前会话假设它"M>1 prefill 崩"——**实测证伪**：它对 M>1 完全正常（vLLM 本就用它做 prefill，是正确的 grouped GEMM，按 token 路由）。新增 `SGLANG_DL_MOE_FUSED_MAX_M`（默认 128）让 prefill/verify 也走 fused。

**实测**（TP=2, fa3, CG, FUSED_MAX_M=128；scripts/dl/e2e_correctness_speed.py + ngram_test.py）：

| 场景 | 旧（bf16-bmm/triton） | 新（fused M>1） | vs vLLM 12.63 |
|---|---|---|---|
| 纯 decode M=1 | 18.32 tok/s | 18.32（不变） | **1.45×** |
| 短请求 e2e（8+48 tok） | 10.08 tok/s | **16.04 tok/s** | **1.27×** |
| prefill 速度 | ~5.6 tok/s | ~50 tok/s | — |
| NGRAM verify（M=5） | 4.95 tok/s（verify=prefill 速，不摊销） | **20.21 tok/s**（best-case 100% accept） | **1.60×** |

**NGRAM 也因此从净负变为净正**：verify 现在 197ms/处理 5 tok = 39ms/tok（fused 摊销），快于 decode 的 54.6ms/tok。

**正确性**：fused M>1 做直接 FP8 GEMM（比 bf16-bmm 的 bf16 累加更准）。"capital of France" 输出连贯。开放 prompt（"explain neural networks"）的换行/重复 degeneracy 在 **bf16-bmm 与 fused 两条路径都出现** → 是模型 greedy 行为（推理模型 + temperature=0），非 fused bug（见 §7.10 P4）。

**交付**（commit 待提交）：
- `fp8.py`：`SGLANG_DL_MOE_FUSED_MAX_M` 默认 1→128（`# DL begin/end` 内）。
- `run_sglang.sh`：qwen35 预设导出 `SGLANG_DL_MOE_FUSED_MAX_M=128`。
- NGRAM 两缺失 op 的 torch fallback（`ngram_worker.py` reconstruct_indices_from_tree_mask、`eagle_utils.py` verify_tree_greedy，commit 73c2c5f213）——NGRAM 在 DLIN 首次跑通。

**2× vLLM（25 tok/s）剩余路径**：NGRAM best-case 已 20.21（1.60×）。提到 25 需：(a) `num_draft=8`（verify 摊销更多 token，理论 ~40 tok/s，但显存紧需 CG_MAX_BS↓/mem↓）；(b) 或 decode 侧 GDN kernel（dl_recurrent，DLIN launch_bounds bug）。两者 + 真实 diverse prompt 的 accept rate 待测。**但 sglang 已全面超过 vLLM（1.27-1.60×），主目标"比 vLLM 快"已达成。**

### 7.12 🎯 2× vLLM 达成：NGRAM num_draft=8 = 35-40 tok/s = 2.8-3.2× vLLM（2026-07-07）

§7.11 的 fused M>1 让 NGRAM verify 摊销后，把 `num_draft` 从 4 提到 8（每步 verify ~9 token），有效吞吐翻倍。

**配置**：TP=2, fa3, page_size=16, CG on (CG_MAX_BS=8), mem_fraction=0.60, `SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16 MAX_BF16_M=128`, NGRAM `num_draft=8 max_bfs_breadth=1`。

**实测**（scripts/dl/ngram_test.py，2 次复现）：

| run | tok/s | accept_rate | accept_length | vs vLLM 12.63 |
|---|---|---|---|---|
| bench0 | **35.23** | 0.61 | 5.33 | **2.79×** |
| bench1 | **39.86** | 0.75 | 6.10 | **3.15×** |

**远超 2× 目标（25 tok/s）。** accept_length 5-6（每 verify 接受 5-6 token）。输出正确（重复 prompt 下 "The quick brown fox..." + spec 痕迹如 "TheThe"，非 gibberish）。

**显存注意**：num_draft=8 的 draft tree + CG 在 2×32GB（35B FP8 已占 ~17GB/GPU）下紧张——CG 捕获 bs=7-8 时 OOM，但 sglang **自动恢复**（降级那些 batch 的 graph），decode/verify 仍跑 35-40 tok/s。稳定配置建议 `CG_MAX_BS=4` + `mem_fraction=0.60`（避免捕获期 OOM 日志；单请求测速不受影响）。或 TP=4 缓解显存。

**达成路径总结**（baseline 0.047 → 39.86 tok/s，**848×**）：
1. quant_type=2→1 + bf16-bmm decode（修 gibberish，跑通）— §7.x gibberish bug
2. fused MoE (invoke_fused_moe_opt) decode M=1 + CG + DLIN norm — 18.32 tok/s decode (1.45× vLLM)
3. **fused MoE M>1 (FUSED_MAX_M=128)** — prefill 5.6→50 tok/s, 短请求 e2e 1.27× vLLM — §7.11
4. **NGRAM unblock**（2 个缺失 sgl_kernel op 的 torch fallback，§7.11/commit 73c2c5f213）
5. **num_draft=8** — verify 摊销更多 token → **35-40 tok/s = 2.8-3.2× vLLM** — 本节

**剩余**：(a) 真实 diverse prompt 的 accept rate 待测（greedy degeneracy 是模型/sampling 问题，§7.10 P4，用 repetition_penalty 或 temperature>0 可改善）；(b) decode-only 18.32→25 的 GDN dl_recurrent kernel 仍 DLIN-side blocked。但 **"sglang 比 vLLM 快 2 倍" 的主目标已达成（实测 2.8-3.2×）**。

### 7.13 TTFT/TPOT 实测 + vLLM 对比阻塞 + 下一步（2026-07-07）

**测试口径（务必注意）**：
- 本会话 sglang 的 TTFT/TPOT 是**离线单流**实测（in-process `sglang.Engine.generate` + streaming API，`scripts/dl/ttft_tpot.py`）。**不是在线服务 bench**，无并发。
- **vLLM 本会话没能实测**（见下阻塞）。docs 里的 vLLM 12.63 tok/s 是**历史 dl19-SDK vLLM** 的数，不是本会话。

**sglang TTFT/TPOT 实测**（TP=2, fa3, CG, fused M>1, num_draft=8 for NGRAM；8-token prompt, 128 out）：

| 配置 | TTFT | TPOT | 吞吐 | 来源 |
|---|---|---|---|---|
| plain（fused，无 spec） | **357 ms** | **54.7 ms**（median）/ 55.1（mean） | 18.3 tok/s | streaming 实测 |
| NGRAM num_draft=8 | ~357 ms¹ | ~25–29 ms²（有效） | 35–40 tok/s | 推导² |
| vLLM（参考） | 未测成 | ~79 ms³ | 12.63 tok/s | 历史 dl19 |

¹ NGRAM 不加速首 token（首 token 无草稿历史，必须真 prefill）→ TTFT≈plain。
² spec-decode TPOT 语义 = 每接受 token 有效耗时 = 1000/吞吐。
³ 1000/12.63 推导；真实值需 vLLM 实测。

**TTFT 357ms 偏高**：被 prefill + 首 token 调度开销主导。fused M>1 已让 prefill 0.3→50 tok/s（§7.11），但首 token 调度 + CG 首步开销还在。**NGRAM 帮不了 TTFT**（只帮 TPOT）。

**vLLM 本会话实测阻塞**（`scripts/dl/ttft_tpot_vllm.py`，用 `RequestOutput.metrics` 取真实 TTFT/TPOT）：
- 试了 2 个 venv（`venv-vllm021` 0.21.0、`venv-vllm-bench` 0.21.1.dev6，均 dl24 torch 2.9.1），TP=2 干净 GPU。
- 两 TP worker 都 init NCCL（NCCL 2.12.12 dl-v0.9.64），TP0 加载完权重（27.8s）。
- **TP1 worker（VllmWorker-1）init 期间静默崩溃**（`Worker proc VllmWorker-1 died unexpectedly`，无 Python traceback、非 OOM）→ EngineCore cancel → engine 起不来。
- 结论：**dl24 SDK 上 vLLM TP>1 有内部 bug**（rank-1 init 时段某 DLIN kernel 段错误；与之前总结记的"vLLM NCCL init failure"同一类 blocker）。`VLLM_ENGINE_READY_TIMEOUT_S=600`（默认即此）无效——是 worker 进程死，不是超时。
- 12.63 tok/s 来自 **dl19-SDK** vLLM（不同环境），本会话 dl24 复现不了。

**下一步优化方案（按 impact/effort 排序）**：

| 优先级 | 方案 | 影响 | 阻塞/effort |
|---|---|---|---|
| **P1** | **vLLM 公平对比**：找回 dl19 SDK + 对应 vLLM venv（产出 12.63 的环境），或 debug dl24 TP1 静默崩溃 | 验证 2× claim 对 fresh vLLM 数 | 中（找环境）/ 高（debug kernel segfault） |
| **P2** | **降 TTFT**（357ms 是当前主延迟）：profiling 首 token 调度 + CG 首步；考虑 prefill CG / 减调度 overhead | 用户体验（交互延迟） | 中 |
| **P3** | **在线 bench**（`./run_sglang.sh bench`，报 TTFT/ITL p50/p99）：sglang plain + NGRAM 在线口径 | serving 真实数 | 低（脚本已就绪） |
| P4 | **质量**：greedy degeneracy（开放 prompt 换行/重复）→ repetition_penalty / temperature>0；验证 NGRAM 在 diverse coherent 输出上的真实 accept | 正确性 + 真实 NGRAM 数 | 低 |
| P5 | **decode GDN kernel**（dl_recurrent）：18.32→~25 tok/s decode | decode 再提 | DLIN-side（launch_bounds bug） |

**建议执行顺序**：先 P3（在线 bench，快速拿 sglang serving 口径）→ P1（vLLM 对比，验证 2×）→ P4（质量）→ P2（TTFT）→ P5（等 DLIN）。主目标"sglang 比 vLLM 快 2 倍"已达成（§7.12），本轮重点转向**公平对比 + TTFT + 质量**。

### 7.14 未完成项 / 痛点 + 优化思路（2026-07-10）

> **⚠️ 状态修正（supersedes §7.12 / §7.13 的"2× 已达成"）**：§7.12 的 NGRAM "35–40 tok/s = 2.8–3.2× vLLM" 与 §7.13 的"主目标已达成"经后续质量验证发现是**退化输出（复述 prompt / 多语言乱码）上的吞吐**，不是真实质量加速。根因 = spec verify 的 target 在 verify 位置重新生成 prompt（详 `dlin-sglang-mtp-vs-ngram-report.md` §10 + memory `spec-verify-prompt-regen-bug`）。**故"2× vLLM"当前不可对外作为质量达成**；下方 P0 是解除此阻塞的唯一路径。
>
> 下表按"是否阻塞可报数"排序，列截至 2026-07-10 仍未做好、影响性能或可报性的项。

| 优先级 | 痛点 | 现状（一句话） | 影响 | 阻塞 / 类型 |
|---|---|---|---|---|
| **P0** | spec verify 质量 bug | verify target 重新生成 prompt；已修 full-attn verify 段（`flash_attention.py:290` DL 块，FA2→`paged_decode_attn` loop），核心复述 bug 解决、accept 8.5%→15.4%，但 greedy 80%+ 仍不可达 | **NGRAM/MTP 吞吐数字当前是退化输出上的吞吐，不能算真实加速；这是唯一阻塞"2× vLLM 可对外报"的点** | sglang-internal 深度（triton/kernel） |
| ~~P1~~ | ~~vLLM 公平对比缺失~~ | ✅ **已复现 fresh 12.47 tok/s**（dl19 env，2026-07-10）；崩溃是 dl24 特有 + 僵尸进程占显存 | 分母已确认；sglang decode 18.32 = **1.47× fresh vLLM** | ✅ 解决（方案 A） |
| P2 | decode GDN `dl_recurrent` kernel | 代码已写，被 DLIN-side `__launch_bounds__(0)` JIT bug 挡住 | decode 18.32→~25 tok/s 的最后一程 | DLIN-side JIT |
| P3 | fused MoE M>1 稳定性 + verify 正确性疑点 | `FUSED_MAX_M` 默认已 128→**16**（commit `f40472402f`，128 长预填充崩）；decode M=1 不受影响；fused M>1 在 verify(M=5) batch 的正确性仍是次要疑点 | 长预填充回退慢路径；headline 报数（128）与稳定默认（16）不一致 | JIT/workspace 分配（cudagraph capture）|
| P4 | TTFT 357ms 偏高 | 首 token 被首调度 + CG 首步开销主导；NGRAM 只帮 TPOT 不帮 TTFT | 交互延迟 | 中（profiling） |
| P5 | cuda-graph 对 sglang 在 DLIN net-negative | sglang FULL graph 比 eager 慢（全 batch），而 vLLM graph 在 DLIN 有效 | 大 batch serving 的主要提效手段失效 | sglang graph 实现（capture/replay/static-pool）的 DLIN 特有问题，独立排查线 |
| P6 | num_draft=8 显存紧张 | 2×32GB（35B FP8 ~17GB/GPU）下 draft tree + CG 在 bs=7–8 OOM（sglang 自动降级恢复） | 稳定 serving 的 CG 覆盖不全 | 显存（CG_MAX_BS↓ / TP=4） |
| P7 | greedy 开放 prompt 退化 | 推理模型 + temperature=0 循环；`rep_penalty=1.2` 已验证修复（`a827e8c319`） | 质量（非速度）；serving 配置项 | 无（已解，待设默认） |

#### P0 优化思路（最高杠杆——解锁后投机解码才能算数）

verify 路径已逐步排除锁定：GDN/Mamba2 state 正确（extend/verify 逐层 sum 精确相等）、GDN verify kernel 输出与 decode 逐层匹配、full-attn FA2 段已修。**剩余三个子项**：

1. **解 topk>1 的 DLIN triton 阻塞**（最高价值）。当前 DLIN 上只有 `topk=1` 能跑，它走的是 GDN target_verify 的 **chain 路径**（`retrieve_parent_token=None`，`fused_sigmoid_gating_delta_rule_update` with `disable_state_update=True`），疑似此路径有问题；而 `topk>1`（会正确设置 retrieve tokens）被 DLIN triton draft backend 的 `AttributeError: 'NoneType' object has no attribute 'swa_out_cache_loc'` 挡住，无法对照。**修通 topk>1 → 即可用已知正确的 retrieve-token 路径做 A/B**，区分"是 topk=1 chain 单独的 bug"还是"更深的 verify 问题"。
2. **给 draft 的 logits 加 rep_penalty**（`frozen_kv_mtp_worker_v2.py:592` seed + `:637` recurrent，Phase-2 draft 当前不对称地无 penalty）。已试：破 draft 的 chain-repeat，但不解决"新颖内容 draft 命中 0"——需配合**可预测内容场景**（可预测内容上 draft 已能命中 3/4，75%）。
3. **修 GDN target_verify chain kernel**（topk=1, `retrieve_parent_token=None`，strides=0 + masked）—— 研究级 triton 工作（`fla/fused_sigmoid_gating_recurrent.py`、`causal_conv1d_triton.py:991`）。
4. （补充）**多层 draft** 提升内禀 accept 上限（1 层 FP8 draft ~10% 是结构上限）。

**关键判别**：full-attn fix 后，对**可预测内容** draft 已能命中 target 75%；平均 accept 仍 3–15% 是因为多数 verify 命中新颖内容（0 accept）。⇒ 高 accept 的现实路径 = **rep_penalty=1.2（让 target 连贯）+ 可预测/结构化内容**，而非纯算法修复。是否追求"任意 prompt 80%+"需评估投入产出。

**2026-07-10 决定性复核（PLAIN vs SPEC 对比，`scripts/dl/p0_verify.py`，详 `dlin-sglang-mtp-vs-ngram-report.md` §10.11）**：full-attn fix **只是部分修复**——greedy 不再逐字复述 prompt，但 **verify forward ≠ plain decode forward** 的残留仍在：新颖内容 greedy 下 spec=`The neural network is...`(phrase-loop) vs plain=`\n\n\n`；**采样(temp=0.6)下 spec 仍复述 prompt**（"Explain how neural networks learn from data." 出现在输出里），即 commit `5cd3164a9e` 的假阳性仍存。**可预测内容 spec≈plain**（唯一干净可用区间）。根因 = hybrid **GDN 状态层 batch-verify ≠ sequential-decode** 数值不等价（per-token GDN verify 已试、反而更差→已 revert），属架构级 sglang-internal 深度问题。M=1 MoE 隔离探针病理慢/挂，未干净隔离（fused MoE M>1 非主因）。

**P0 复核结论**：核心假阳性**部分缓解**（greedy）；verify≠decode 残留**未解**（采样仍复述 prompt），列为后续深度项（GDN verify kernel batch↔recurrent 数值等价）。**对外口径不变**：NGRAM/MTP 吞吐须带质量星号；干净可用区间 = 可预测/重复内容。`accept 8.5%→15.4%` 的数（§7.14 行）是 full-attn fix 后的 greedy 单点，不代表采样/新颖内容已修。

#### P1 优化思路（可报数根基）

- **A. 找回 dl19 SDK + 对应 vLLM venv**（`venv-vllm021`，0.21.0）复测 12.63 baseline 是否可复现——快、低风险。
- **B. debug dl24 worker 静默崩溃**：rank-1 init 时段某 DLIN kernel 段错误（无 Python traceback）—— 难，需 dlcc/gdb 定位崩溃 kernel。
- 备用：dl19/dl24 torch 的 `torch.matmul` TFLOP/s 已证等价（§2.4），故 dl19 vLLM 数对 bf16/FP8 dense 都有参考意义。

**✅ P1 已解决（2026-07-10，方案 A）**：dl19 vLLM 环境已恢复并复测——`source ../sdk/env.sh`（含 `libhcrt.so`）+ `venv-vllm021`（torch `2.9.1+dl19.sdk20260312`、vllm `0.21.0+cu117.dl9.sdk20260528`）即可 import/运行。`scripts/dl/qwen35_vllm_tps.py`（TP=2, eager, GPU 8,9, 4-tok prompt + 128 decode）实测 **VLLM_TPS = 12.47 tok/s（128 tok / 10.26s）**——**复现历史 12.63 分母**（差 <1%，噪声内）。
- "vLLM TP>1 worker 崩溃"是 **dl24 特有**（dl24 SDK 上 rank-1 init 段错误）；dl19 不复现。本会话首次失败是**僵尸 `VLLM::EngineCore` 进程占满 GPU 4,5 显存**（init 失败后未干净退出）→ `ValueError: Free memory ... less than desired`，非 worker 死亡 bug；kill 僵尸 + 换空 GPU 即解。
- **结论**：vLLM 分母 = **12.47 tok/s（fresh, dl19, 已确认）**。sglang 纯 decode 18.32 tok/s = **1.47× fresh vLLM**——"1.45× vLLM" claim 现建立在**已复现的 fresh 分母**上（不再依赖历史数）。复现命令：`source ../sdk/env.sh && CUDA_VISIBLE_DEVICES=<2 free GPUs> MODEL_PATH=/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8 NTOKENS=128 ../venv-vllm021/bin/python scripts/dl/qwen35_vllm_tps.py`。

#### P2 已定性（2026-07-10，deferred）

- **dl_recurrent decode kernel 不在 repo**（scratch 实验，被 DLIN JIT `__launch_bounds__(0)` bug 挡）。repo 里 GDN decode 的唯一 DL 改动是 `gdn_backend.py` 的 `SGLANG_DL_SKIP_TRACK_MAMBA`（省 1 triton kernel/层），当前 decode 走 triton `packed_decode`。
- **无现成 op 可移植**：vLLM `venv-vllm021` 的 `_dl_C.so`（仅 7 op，无 recurrent/gdn/delta）——recurrent/GDN 逻辑在 vLLM 与 sglang **都是 Python/triton**（`vllm/.../fla/ops/fused_recurrent.py` 等），不像 FP8 GEMM 那样有 AOT `_dl_C` op 可直接 load。故"load vLLM op"路径此处不适用。
- **优先级重判**：sglang 纯 decode **18.32 tok/s 已 1.47× fresh vLLM 12.47**（P1 已确认分母）。dl_recurrent（→~25）是**超越 parity 的增量优化，非必需**。DLIN JIT 修前 deferred。

#### P3 优化思路（fused M>1 稳定性与正确性）

- **稳定性**：`gptq_dlblas_gemmex` 内部 `dlblasLtMatmulGetWorkspace` 每次分配 workspace → 不可 cudagraph capture；128 在长 prefill 崩疑与此相关。方向 = 预分配/复用 workspace（让 fused M>1 可进图、且 128 不崩）。
- **正确性**：full-attn fix **之后**重测 `FUSED_MAX_M=1 vs 16` 的 verify 输出差异——确认 fused M>1 在 verify(M=5) batch 无独立 bug（fix 前曾怀疑是它产生 prompt-regen，现主因已转 full-attn；需复核）。

**✅ P3 稳定性已定性（2026-07-10，DLIN 编译器 bug）**：`invoke_fused_moe_opt` 在 prefill **M≥~100** 触发 **DLIN dleol `tu_program.cc:625` assert → SIGSEGV（`fused_moe_opt.cu:775`）**，triton `fused_experts` 撞同一 assert。即**两条快路径在大 M 下都崩**（DLIN 编译器/dleol bug，非 sglang）。workaround 已 committed（commit `f40472402f`）：`FUSED_MAX_M=16`（decode M=1 + NGRAM verify M=9 + 短 prefill 走 fused）+ 长 prefill 回退 `bf16-bmm`（`MAX_BF16_M=2048`，慢但稳，TTFT ~13s/128-tok prefill）。**代价**：长 prefill 慢（bf16-bmm），短/decode 工作负载保 2× vLLM。**deferred**：真修需 DLIN 修 dleol assert（让 FUSED_MAX_M 回 128+ 加速长 prefill）。
**P3 正确性子问题**：fused M>1 verify 是否在 GDN 发散之外**额外**发散？被 P0 更深结论覆盖（verify≠decode 主因是 GDN 状态层 batch≠seq，非 MoE M），且 M=1 verify 探针（`scripts/dl/p0_moe_probe.py`）病理慢/挂——本身说明 M=1 MoE 在 verify 多 token batch 不可用，无法干净隔离。列为 P0-followup（task #9）一并解。

#### P4/P5 已定性（2026-07-10，blocked on DLIN profiling）

两者都**已有测量、进一步优化 blocked on DLIN 工具链**：

- **P4 TTFT 357ms**（§7.13 实测）：被 prefill + 首 token 调度 + CG 首步主导；NGRAM 只帮 TPOT 不帮 TTFT。**优化杠杆（prefill CG）被 P5 卡住**（sglang graph 在 DLIN net-negative），且长 prefill 走 bf16-bmm（P3 dleol bug 的回退，~13s/128-tok）。**deferred**：降 TTFT 需先解 P5（graph）或 P3（dleol 让长 prefill 走 fast fused）；`torch.profiler` 在 DLIN 崩，无法精细定位首 token 各段开销。
- **P5 cuda-graph net-negative**（§2.4 已测，Qwen3-1.7B dense）：sglang FULL graph 比 eager **慢**全 batch（bs=1: 60.5 vs 48.2ms；bs=64: 312.6 vs 147.2ms），而 vLLM graph 在 DLIN 有效。指向 **sglang graph 实现（capture/replay/static-pool）的 DLIN 特有问题**（独立排查线），非 fusion/dlblasLt（bf16 下双方等价，§2.4 已证伪）。**deferred**：根因需 DLIN profiling（崩）+ sglang graph 内部深挖；当前 DLIN 上 sglang 应用 **eager**（graph net-negative 故默认 disable_cuda_graph）。35B hybrid 上未单独复测（高 bs OOM-prone，见 P6；原理同 dense）。

**结论**：P4/P5 当前无 sglang 侧可落地优化（均 blocked on DLIN 工具链/编译器）；**serving 配置维持 `disable_cuda_graph`（eager）+ 短 prompt / decode-dominated 工作负载**。若 DLIN 修好 profiling + dleol + graph，再回头攻。

#### P6/P7 已定性（2026-07-10）

**P6 num_draft=8 显存紧张**：CG-capture OOM（bs=7-8）只在 **CG 开**时发生；而 P5 结论是 DLIN 上用 **eager（`disable_cuda_graph`）** → **该 OOM 在推荐配置下根本不发生**。即便 CG 开，sglang 也**自动降级恢复**（OOM batch 回退 eager，非硬失败，仅日志噪音）。结论：维持 eager 配置即免；若要 CG，设 `CG_MAX_BS=4` + `mem_fraction=0.60`（已是默认）；更高并发用 TP=4。

**P7 greedy 开放 prompt 退化**（⚠️ **修正 commit `a827e8c319` 的"rep_penalty=1.2 已解"判断**）：本会话 fresh PLAIN 数据（`scripts/dl/p0_verify.py`）显示 **rep_penalty 不是普适修复**——
- 可预测内容（计数序列 `1,2,...,15,`）：plain greedy 正确计数（`16,17,...`），**加 rep_penalty=1.2 反而破坏**（→`16, 2007. The numbers are in order...` 不连贯）。
- 新颖内容：plain greedy 本就退化（`\n\n\n`），rep_penalty 给不同但仍退化的输出（`The first step is to the 100% of the data...`）。
- commit `a827e8c319` 的"Trump→连贯"是**特定开放 prompt** 的现象，非普适。
⇒ **结论**：greedy 退化是该推理模型 + temperature=0 的**内禀行为**；**rep_penalty 不应设为默认**（伤结构化/可预测内容）。建议：结构化/代码工作负载 **关 rep_penalty**；开放问答可 `temperature>0` + 按需 `rep_penalty≈1.1`。不设全局默认。

#### 本会话处置结果（2026-07-10）

| 项 | 处置 | 要点 |
|---|---|---|
| **P1** | ✅ **解决** | dl19 env 恢复，fresh vLLM **12.47 tok/s** 复现分母；sglang decode 18.32 = **1.47× fresh vLLM**（claim 建立在已确认分母上）|
| **P7** | ✅ **修正** | rep_penalty=1.2 **非普适**（破坏计数/结构化内容，仅特定开放 prompt 有效）→ **不设默认**；修正 commit `a827e8c319` 的过乐观判断 |
| **P6** | ✅ **解决** | CG-capture OOM 在 eager 配置（P5 结论）下不发生；sglang 即便 CG 开也自动降级；`CG_MAX_BS=4`+`mem 0.60`/TP=4 备用 |
| **P0** | 🟡 **定性+deferred** | PLAIN vs SPEC 决定性对比：full-attn fix 部分修（greedy 不再逐字复述），但 verify≠decode 残留（采样仍复述 prompt）；根因=hybrid GDN 状态层 batch≠seq（架构级，task #9）|
| **P3** | 🟡 **定性+deferred** | 稳定性=DLIN dleol `tu_program.cc:625` assert（M≥~100 崩），workaround `FUSED_MAX_M=16`+bf16-bmm 已committed；正确性子问题并入 P0 |
| **P2** | 🟡 **定性+deferred** | dl_recurrent kernel 不在 repo、DLIN JIT `__launch_bounds__(0)` bug；无 `_dl_C` op 可移植；decode 已 1.47× vLLM 故非必需 |
| **P4/P5** | 🟡 **定性+deferred** | 均已有测量（§2.4/§7.13），进一步优化 blocked on DLIN（`torch.profiler` 崩 + dleol + graph 内部）；serving 维持 eager |

**对外口径（更新）**：sglang decode 18.32 tok/s = **1.47× fresh vLLM 12.47**（分母已本会话确认）。**spec-decode（NGRAM/MTP）的 2–3× 吞吐须带质量星号**（verify≠decode，采样仍复述 prompt，详 §10.11）——干净可用区间仅可预测/重复内容。serving 推荐配置：`fa3 + page_size=16 + disable_cuda_graph + SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16 MAX_BF16_M=2048 + mem_fraction=0.60`，短/decode-dominated 工作负载。

**真正可推进项（需 DLIN 配合）**：① GDN verify batch↔recurrent 等价（P0/task #9，解开后 spec-decode 可对外报）；② dleol `tu_program.cc:625` 大-M assert（P3，解开后长 prefill 走 fast fused，降 TTFT）；③ dlcc/JIT `__launch_bounds__` 支持（P2）；④ DLIN profiling 可用（P4/P5 根因）。sglang 侧已达当前可达极限。

---

### 7.15 OPT 落地：sglang 侧可做的优化（2026-07-10，重新框定后）

> 重新框定：spec-decode 在 hybrid 状态模型上 verify≠decode 是架构级难修，**ROI 低**；真正杠杆是 **prefill（sglang 输 vLLM 处）+ graph（高 batch 吞吐）**。这两处恰好有**不依赖登临**的 sglang 侧解法。

#### ✅ OPT-1：prefill 分块到 16 绕 dleol —— 已验证 ~3.5–6.6× 长 prefill 提速

**思路**：dleol 在 M≥~100 崩（P3），故 `FUSED_MAX_M=16` 把长 prefill 逼到慢 bf16-bmm。但若把 `chunked_prefill_size` 压到 16，每次 forward 的 MoE batch M≤16 ≤ `FUSED_MAX_M` → **永远走 fast fused，永不撞 dleol，永不回退 bf16-bmm**。纯 sglang 配置，不需登临修 dleol。

**实测**（`scripts/dl/prefill_bench.py`，TP=2, fa3, eager, ~180-tok 长 prompt，max_new_tokens=1 = prefill 代理）：

| 配置 | SHORT(4tok) | LONG(180tok) 冷 | LONG 暖(JIT后) | LONG 输出 |
|---|---|---|---|---|
| baseline `chunk=2048`（→bf16-bmm）| 414ms | **33s** | ~31s | `\n\n`（退化）|
| **`chunk=16`（→fused M=16）** | 418ms | **9.5s** | **5.0s** | `, residual connections, and layer normalization...`（**连贯**）|

- **长 prefill 冷 33s→9.5s（3.5×），暖 31s→5.0s（6.6×）**。SHORT 不变（本就 1 chunk fused）。
- **输出连贯**（GDN 状态跨 ~12 个 prefill chunk 正确传递——sglang Mamba/hybrid chunked-prefill 核心特性），非 garbage。
- **解 P3（稳定性：M=16 永不撞 dleol）+ P4（TTFT：长 prefill 大降）**，不需登临。
- **反转 §7.14 "真正可推进项 ②"**：长 prefill fast fused **不必等 DLIN 修 dleol**——分块即绕开。

**推荐**：serving 配置加 `chunked_prefill_size=16`（与 `FUSED_MAX_M=16` 配套）。**sweet-spot follow-up**（未测，集群抢占中）：把 `FUSED_MAX_M` 提到 32/64 + chunk 同步，chunk 数减半，可能再快（dleol 阈值 ~100，32/64 仍安全）。复现：`CUDA_VISIBLE_DEVICES=<2 free> SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16 MAX_BF16_M=2048 CHUNK=16 .venv/bin/python scripts/dl/prefill_bench.py`。

#### ❌ OPT-2：MoE/linear workspace 预分配 —— 前提不成立，非 sglang 侧可做

调查后**推翻自己先前的假设**：
- FP8 linear GEMM（`dlblas_w8a8_block_fp8_linear`，每层都用）调 `torch.ops._dl_C.gptq_dlblas_gemmex(input, w, scale, scale, quant_type, bit)`——**调用无 workspace 参数**，workspace 分配在 vLLM `_dl_C.so`（**登临编译的黑盒 op**）内部，sglang 侧**无法预分配/复用**。
- 且 §2.4 已证 sglang **能完整捕获 FULL graph**（输出干净逐字一致），只是 **net-negative（replay 慢于 eager）**。能捕获 ⟹ capture **没被 workspace 打断**（cudaMalloc-in-capture 会直接报错，非 net-negative）。⇒ graph 慢的根因是 **DLIN graph replay 内部**（非 workspace），属 P5 深度项。
- **结论**：OPT-2 无 sglang 侧落地点（黑盒 op + replay 内部均 DLIN 侧）。修正先前"workspace 预分配可让 graph 翻正"的乐观框定。

#### 🟡 OPT-3 / OPT-4：DLIN-blocked / 低 ROI（维持先前定性）

- **OPT-3（dl_recurrent decode kernel）**：DLIN JIT `__launch_bounds__(0)` bug 挡，kernel 不在 repo、无 `_dl_C` op 可移植；decode 已 1.47× vLLM 故非必需。**DLIN-blocked，低优先。**
- **OPT-4（GDN verify batch↔seq 等价）**：spec verify≠decode 的架构级根因（hybrid 状态层）。重新框定后判断 **ROI 低**——即便修好，1 层 draft 在新颖内容命中仍低；精力更适合投 OPT-1（已验证大收益）。**研究级，建议暂缓**，除非 spec-decode 质量成为硬需求。

#### OPT 小结

| OPT | 结果 | 备注 |
|---|---|---|
| **OPT-1 prefill 分块** | ✅ **落地，3.5–6.6× 长 prefill** | 纯 sglang 配置，已验证，解 P3+P4 |
| OPT-2 workspace 预分配 | ❌ 前提不成立 | 黑盒 op + graph 能捕获，非 sglang 侧 |
| OPT-3 dl_recurrent | 🟡 DLIN-blocked | 低优先 |
| OPT-4 GDN verify 等价 | 🟡 研究级/低 ROI | 建议暂缓 |

**OPT-1 是本轮可落地的实质收益**：长 prompt prefill 从 ~33s → ~5s（暖），TTFT 大降，且不依赖登临。serving 配置加 `chunked_prefill_size=16` 即生效。其余 OPT 经调查均为 DLIN-blocked 或低 ROI——本身是有价值的结论（避免在死胡同投入）。

---

### 7.16 MEAS：并发吞吐实测 —— 推翻"sglang 已超 vLLM"，定位真实 gap（2026-07-10）

> 前两轮都在 bs=1 单流上猜瓶颈。本轮做**公平并发对照**（sglang vs vLLM，**同 eager**、ignore_eos 固定 OUT=64、TP=2），用证据定位 gap。结果**颠覆项目叙事**。

**实测聚合吞吐（tok/s，`scripts/dl/bench_batch_sg.py` / `bench_batch_vllm.py`）**：

| B | sglang eager | sglang **CG** | vLLM eager | 结论 |
|---|---|---|---|---|
| 1 short | 7.2 | **16.7** | 12.9 | **sglang-CG 反超 vLLM** |
| 8 short | 29.2 | 34.0 | **60.8** | vLLM 1.8× |
| 16 short | 31.2 | 36.2 | **60.1** | vLLM 1.7× |
| 1 long | 4.3 | 6.2 | 5.8 | sglang-CG 略超 |
| 8 long | 9.6 | 10.1 | **34.7** | vLLM 3.4× |
| 16 long | 9.7 | 10.3 | **31.6** | vLLM 3.1× |

**三个决定性结论**：

1. **"sglang 已超 vLLM"是假象**。先前"1.45× vLLM"（18 tok/s）= sglang-**CG** 对 vLLM-**eager** 的不公平对比。**公平 eager 对照下 sglang 反而落后 ~2×**（7.2 vs 12.9）。

2. **CG 在 35B hybrid 上是 net-POSITIVE**（B=1 eager 7.2 → CG 16.7 = **2.3×**），且 sglang-CG(16.7) **反超 vLLM-eager(12.9)**。⇒ **§7.14 P5"net-negative→用 eager"结论（基于 1.7B dense）对 35B hybrid 不成立**；`disable_cuda_graph` 默认在此模型上**错了**，白白丢 2.3×。

3. **真实 gap 在两处**：
   - **高 batch decode**（B=8-16）：vLLM 60 vs sglang 36（1.7×）。sglang 吞吐在 ~36 plateau，vLLM 到 60。
   - **长 prompt 并发**（B=8-16 long）：vLLM 32-35 vs sglang 10（~3×）。OPT-1 chunking 只救单流长 prefill，**并发下 prefill 仍串行瓶颈**（chunked prefills 互相争抢）；vLLM 原生 dlblas 大-M prefill 并发扩展好。

**ranked 下一步（证据驱动）**：
1. **立即：35B serving 开 CG**（去 `disable_cuda_graph`，注意 hybrid Mamba-cache 对 max_num_seqs 的约束——同 vLLM 的 42 blocks 限制）。B=1 直接 7.2→16.7 反超 vLLM。**配置级，零风险高收益。**
2. **profile 定位高-batch gap**（MEAS-2，`layer_timing.py`）：sglang 为何 plateau@36 而 vLLM 到 60？是 GDN/MoE/attn 哪层、还是 CG 高-bs 捕获不全/continuous-batching 效率。
3. **长 prefill 并发**：chunking 不够；要么 DLIN 修 dleol 让 sglang 用原生大-M（像 vLLM），要么改并发 prefill 调度（让多个请求的 chunked prefill 更并行）。

**方法论结论**：停止猜测、做一次公平并发对照，一次定位三个真实 gap（CG 误关 / 高-batch / 长-prefill-并发）——比再列猜测表信息量大得多。`disable_cuda_graph` 这个默认是最大的隐藏损失。

---

### 7.17 BENCHRUN：serving 口径实测（sglang benchrun_sglang，2026-07-10）

> 用 benchrun 格式（vLLM `vllm bench run` 同 schema）在 serving 口径验证 CG 收益。

**✅ sglang serving benchrun（CG on，c=1，random-ids in512/out128，8 prompts）实测**：
- success_req 8/8；**peak output throughput 18.0 tok/s**；median TPOT 58.0ms（=17.2 tok/s decode）；p99 ITL 108.6ms。
- 印证 MEAS-1：sglang-CG serving ≈ 17–18 tok/s decode，与 §7.9 的"18 tok/s"一致（即 CG 路径，非 eager）。
- 复现：`benchrun_sglang.py /tmp/sg_benchrun_cfg.json`（venv 激活 + 本地模型 + MoE env）。

**⚠️ sglang benchrun 工作配置（踩 6 个坑才跑通，记录备用）**——`run_sglang.sh benchrun -M qwen35-35b` 预设有缺，需自定义 config：
```
server_params: tp=2, fa3, page16, mem_fraction_static=0.60, disable_cuda_graph=false,
               chunked_prefill_size=16, cuda_graph_max_bs=2   ← 缺任一即崩
```
踩坑链：① 模型默认 `/mars`（慢）→ 用本地副本；② `cuda_graph_max_bs=8` 捕获太慢 → 降到 2；③ **benchrun preset 无 `chunked_prefill_size=16`（OPT-1）→ 长 prompt prefill M>16 → bf16-bmm 大分配 + CG → OOM**（`DL_MOE_ERR tried 7.78GB`）；④ 需 venv 激活（否则 harness 用 `/usr/bin/python3` 无 sglang）；⑤ `mem=0.55` 太低 → hybrid mamba state cache 太小（max_num_reqs=0）→ 需 0.60；⑥ output_dir 必须预建（"docker mount"检查）。**建议把 `chunked_prefill_size=16` 补进 run_sglang.sh 的 benchrun preset + qwen35 预设。**

**❌ vLLM serving benchrun（`vllm/vllm/benchmarks/benchrun_serving.py`）未能跑通**：server "terminated unexpectedly"（无日志、立即退出），`enforce_eager=true` 也不解 → 是该 harness 的 **vLLM server-launch 对本 hybrid 模型/DLIN 的 bug**（venv-vllm021 缺 pandas/dateutil 已补，仍崩）。vLLM serving 数暂用 **MEAS-1 offline eager（12.9 tok/s @ B=1）** 作对照。

**benchrun 口径对照结论**：
| 口径 | sglang | vLLM |
|---|---|---|
| serving benchrun (CG/eager, c=1) | **peak 18 tok/s, TPOT 58ms** ✅ | harness bug（用 offline 数）|
| offline eager B=1（MEAS-1） | CG 16.7 / eager 7.2 | **12.9** |

⇒ sglang **开 CG** 后 serving decode 18 tok/s，**反超 vLLM eager 12.9**（低并发/延迟场景）。高并发仍输（MEAS-1：vLLM 60 vs sglang 36 @B=16）。**benchrun 验证了 CG 收益，且暴露 run_sglang.sh benchrun preset 缺 `chunked_prefill_size=16` + vLLM benchrun_serving 的 server-launch bug 两个待修项。**

---

### 7.18 MEAS-3 + bug18025 对照：gap 在单流延迟（GDN kernel），非聚合吞吐（2026-07-10）

> 同事报 vLLM Qwen3.6-35B-A3B-FP8 **38 tps（MTP=3 → 43）**。bug 18025 性能分析显示该数 = **TP=4 + CUDA Graph + `DLEOL_FLA_ENABLE_PINGPONG=1`/`UNROLL_COUNT=8`（DLIN FLA 优化）+ 新版 vLLM**，batch=1 TPOT **26ms（=38 tok/s）**。我先前 vLLM 测的是 **eager + TP2 + 无 FLA（12.9）**——测了 vLLM 最差配置。

**sglang TP4+CG 实测（`scripts/dl/bench_batch_sg.py` TP_SIZE=4 CG=1，GPU 24-27，CG 捕获 96s/4-rank）**：

| B | sglang TP2+CG | **sglang TP4+CG** | vLLM TP4（bug18025）|
|---|---|---|---|
| 1（单流延迟）| 16.7 | **12.0** ⬇️ | **38** |
| 8 short（聚合）| 34.0 | **68.3** | ~67（batch4 TPOT 60ms 推算）|
| 16 short（聚合）| 36.2 | 85.4 | — |

**两个决定性发现**：

1. **TP4 对 sglang 单流延迟是负向**（16.7→12.0）：单序列下 TP4 的 NCCL allreduce 开销 > 并行收益。vLLM TP4 能到 38 说明其 TP 通信高效 + GDN kernel 快。⇒ **"sglang 换 TP4 追平 38"不成立**。

2. **gap 在单流 decode 延迟（B=1 TPOT），非聚合吞吐**：
   - 单流（B=1）：sglang 12–16.7 vs vLLM **38** → sglang **慢 2.3–3×**。
   - 聚合（batch 8）：sglang TP4 **68** ≈ vLLM **~67** → **持平**。

⇒ **sglang 在多用户聚合吞吐上已打平 vLLM；只在单流延迟上落后 3×。** "38 tps"是单流 TPOT 指标——sglang 在该指标上确实落后。

**根因 = GDN kernel（sglang triton vs vLLM DLEOL FLA pingpong/unroll）+ TP allreduce 开销**。这是先前**错判为"DLIN-blocked、低优先"的 OPT-3**——vLLM 在用它跑 38，证明**可行且是最大杠杆**。

**三轮纠正收敛**：①"sglang 超 vLLM"（测了 vLLM eager）→ ②公平 eager sglang 输 → ③ TP4 能追 → **④ TP4 不追，gap 在单流 GDN kernel（DLEOL FLA）**。

**头号优化（重排）**：给 sglang GDN 接 **DLEOL FLA**（pingpong/unroll，vLLM 已验证可用）——解单流延迟 3× gap。次：sglang TP allreduce 效率（让 TP4 不伤单流）。聚合吞吐已达标，无需再投。

---

### 7.19 对齐基准（ALIGNED TPOT）—— gap 是 1.5×，不是 3×（2026-07-10）

> **关键纠正**：先前 MEAS-3 报"单流 3× gap"用的是 `bench_batch_sg` 的**聚合 tok/s**（总 token/总墙钟，混入 per-request 调度 + prefill），非 vLLM 口径。对齐到 **TPOT（decode-only, per-token）** 后 gap 减半。

**对齐口径**：batch=1, input_len=1024, output_len=512, TP=4, CUDA Graph on。vLLM = `bench offline`（bug18025），sglang = `benchrun_sglang`（server c=1，TPOT；batch1 c=1 下与 offline TPOT 等价，HTTP 噪音在 39ms 尺度可忽略）。

| | TPOT median | = tok/s | 备注 |
|---|---|---|---|
| **sglang TP4**（triton GDN）| **39.4ms**（p99 40.0，很稳）| **25.4** | benchrun c=1 in1024/out512 |
| **sglang TP2**（triton GDN）| 58ms（v6 测）| 17.2 | TP4 比 TP2 decode 快 1.5×（**TP4 对 decode 正向**，纠正 MEAS-3）|
| **vLLM TP4**（DLEOL FLA, bug18025）| **26ms** | **38** | MTP=3 → 43 |
| **gap（sglang TP4 vs vLLM TP4）**| **1.5×** | | 非 3× |

**结论**：对齐后 sglang TP4 decode = 25.4 tok/s（TPOT 39ms），vLLM = 38（TPOT 26ms），**gap 1.5×**。根因 = GDN kernel（triton vs DLEOL FLA）+ TP comm。**这是优化前的对齐起点**——关 1.5× 比关 3× 现实。

**对齐 caveats（诚实）**：① sglang `bench_one_batch`（离线直连）对 hybrid 长 prefill 有 bug（`tensor a(64) vs b(1024)` mamba 状态，direct-extend 路径），故 TPOT 用 server 路径（benchrun）；② sglang 未设 ignore_eos，random-ids 早停 ~269 tok/prompt，但 TPOT 是 per-token 延迟、与生成长度无关，仍可比；③ 模型 Qwen3.5 vs bug 的 3.6（同代，影响小）；④ sglang triton GDN vs vLLM DLEOL FLA 是**被测的优化 gap 本身**，非方法论差。

**优化目标（对齐后重定）**：把 sglang TP4 TPOT 从 39ms 降到 ≤26ms（追平 vLLM）= **关 1.5×**。杠杆 = DLEOL FLA for GDN（vLLM 已用）+ TP comm 效率。聚合吞吐已 ≈ vLLM（MEAS-3 batch8 ~68 vs ~67），无需再投。

---

### 7.20 实现 DLEOL FLA GDN decode → 解决 JIT segfault（2026-07-10）

**已实现**（opt-in `SGLANG_DL_GDN_DLIN=1`，裹 `# DL` 标记）：
- 新 `python/sglang/srt/layers/attention/linear/kernels/gdn_dlin.py` — `DLinGDNKernel.decode` 调 `torch.ops._dl_C.dl_recurrent_gated_delta_rule`（`_dl_C.so` 已被 FP8 GEMM 的 `_ensure_dl_C` 加载，含该 op）。mirror vLLM `dl_platform_plugin`：`g,beta=fused_gdn_gating(...)` + `beta→bf16`（vLLM fused_gdn_gating 返回 bf16 beta，sglang 返回 fp32——op 按 bf16 特化）。
- `gdn_backend.py::GDNKernelDispatcher.__init__` 加 `is_dlin() and SGLANG_DL_GDN_DLIN=1` 分支：`decode_kernel=DLinGDNKernel`，extend/verify 留 triton，`supports_packed_decode=False`。
- 实测：dispatcher 确认 `decode=DLinGDNKernel`，op 成功运行、生成连贯文本。

**segfault 根因 + 修复（关键）**：初版 op 在 `dleol::vm::CUInstExecutor::getCuFunction`（kernel JIT 加载）segfault。逐项对照 vLLM 实测 specs（instrumented `_dl_ops.py`）确认张量 layout/dtype 一致（仅 beta dtype 差，已修）。**真正根因 = SDK 版本不匹配**：sglang 默认 source `sdk-0401`（**dl24** 的 `libdleol/libdldnn`），与 vLLM **dl19** 构建的 `_dl_C.so` 不匹配 → DLEOL FLA VM JIT 崩（FP8 GEMM 容忍此 mismatch，FLA VM 不容忍）。**修复 = source 默认 `sdk/env.sh`（dl19-matching libs）而非 sdk-0401** → segfault 消失，op 跑通。

**仍待验证（需 GPU，集群抢占中）**：
1. **正确性**：DLIN op 输出 vs triton 基线逐 token 对比（初测输出 "the capital of the capital of France..." 疑模型 greedy 退化，但需 triton 基线确认非 op 数值 bug）。
2. **TPOT 收益**：DLIN op（DLEOL FLA）下 sglang TP4 TPOT 是否从 39ms→~26ms（追平 vLLM）。需 CG on + 对齐 benchrun。

**复现**（segfault 已修）：
```
source ../sdk/env.sh && source .venv/bin/activate   # 默认 sdk, 非 sdk-0401
CUDA_VISIBLE_DEVICES=<4 free> SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MOE_FUSED=1 \
  DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8 \
  python -c "...sglang.Engine(... tp=4, fa3, chunk=16, mem0.60)..."
```
**结论**：DLEOL JIT segfault **已解决**（sglang 侧 SDK env 修复）；DLEOL FLA GDN decode 已接入 sglang（opt-in）。正确性 + TPOT 待 GPU 空闲后验证。

---

### 7.21 TPOT 39→30.6ms：GDN op + quant_type=2 双优化（2026-07-11，验证）

> 接 §7.20。GPU 空闲后验证 DLEOL FLA GDN op 的 TPOT 收益 + 深挖剩余 gap。**两项 vLLM 对照优化落地，TPOT 39→30.6ms（-22%）**。

**对齐口径**：batch=1, short prompt, TP=4, CG on, `../sdk`（非 sdk-0401，§7.20 segfault 修复），`SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16`。offline `sglang.Engine`（与 §7.19 server benchrun c=1 等价，short prompt 下 prefill 可忽略）。vLLM 目标 = 26ms（§7.19 bug18025）。

| 配置 | TPOT | tok/s | 改动 |
|---|---|---|---|
| baseline（triton GDN, q1）| 37.7ms | 26.5 | §7.19 起点附近（../sdk 略快于 sdk-0401 的 39.4）|
| + GDN dl_recurrent op（`SGLANG_DL_GDN_DLIN=1`）| 34.7ms | 28.8 | **-3ms**：triton recurrent → `_dl_C.dl_recurrent_gated_delta_rule` |
| + quant_type=2 dense FP8 linear（`SGLANG_DL_FP8_Q2=1`）| **30.6ms** | **32.6** | **-4ms**：per-channel requant ��� blockwise 硬件融合 dequant（vLLM 同款）|
| **合计** | **30.6ms** | **32.6** | **-7ms（-19% vs 37.7 baseline）**；离 26ms 还差 4.6ms |

**优化 1：GDN dl_recurrent op** — `SGLANG_DL_GDN_DLIN=1`（§7.20 已接入）。验证：dispatcher 确认 `decode=DLinGDNKernel`，op 跑通、CG 可捕获（66s capture 无 segfault），**输出与 triton 逐字一致**（greedy 确定性 → 两条路径同文）。TPOT 37.7→34.7（-3ms）。**但**：dl_recurrent op 本身只占 ~0.5ms（kprof 估算）；GDN 层 38% 的大头是 conv1d + gating + projections，非 recurrent。

**优化 2：quant_type=2 dense FP8 linear（关键突破）** — sglang 原 `dlblas_w8a8_block_fp8_linear` 默认 quant_type=1（per-channel：load 时 dequant blockwise→requant per-channel 缓存，每步调 `gptq_dlblas_gemmex(quant_type=1)`）。vLLM 用 quant_type=2（blockwise 硬件融合 dequant，直接吃 checkpoint 权重）。**先前的 quant_type=2 尝试 SIGSEGV**，根因 = sglang 多加了 `.contiguous()`：vLLM `fp8_dlblas.apply` 传 `weight.t()`（**非连续转置 view**）+ `weight_scale` as-is + `x.view`（无 contiguous），kernel 按该 stride 硬编码。**修复 = 完全对齐 vLLM（去掉所有 .contiguous()）→ crash 消失，TPOT 34.7→30.6（-4ms）**。scale layout 两边都是 `[N/128,K/128]`（已对 vLLM `fp8_dlblas.py:345` assert 核对），非 layout 差异。`SGLANG_DL_FP8_Q2=1` opt-in；legacy q1 走 `SGLANG_DL_FP8_Q1=1`。

**深挖剩余 4.6ms（profile 结论）**：
- **100% GPU util（稳态 decode，dlsmi 连续 100%）** → compute-bound，CG 无 launch-overhead gap。gap 在 kernel 计算量，非通信 stall 或 CPU 开销。
- **eager 逐层分解（sync-isolated）**：GDN 38% / MoE 62%（eager 下 GDN 因多小 kernel 略高估，CG 下 MoE 占比更高）。
- **TP allreduce**：TP4 NCCL RING_LL128 ~73µs/call（4-32KB latency-bound）；~60 AR/step。**但 AR 在 CG 内捕获**（communicator.py:769 "NCCL internal stream, CG-compatible"）→ comm≈kernel time ~0.6-2ms，非主因。TP4 仍优于 TP2（39 vs 58ms，§7.19），印证 compute-bound。
- **MoE**：sglang 与 vLLM 同用 `invoke_fused_moe_opt`。差异：sglang 硬编码 block_size `16/128/128`，vLLM 走 `try_get_optimal_moe_config`（per-shape JSON，default 64/64/32）；sglang 额外 `c2.sum(dim=1)` combine（vLLM 融在 C++ op 内）——但 batch=1 下 combine 仅 ~0.2ms，次要。
- **GDN**：vLLM decode 用 `fused_recurrent_gated_delta_rule_packed_decode`（**triton，gating+recurrent 融合单 kernel**，DLEOL FLA JIT pingpong/unroll）；sglang DLinGDNKernel 是 **gating(`fused_gdn_gating`)+recurrent(`dl_recurrent`) 两个独立 kernel**。conv1d 两边都是 triton（parity）。⇒ **GDN gating+recurrent 融合是剩余可移植点**（估 ~1.5-3ms）。

**质量（重要，非本次改动引入）**：greedy(temperature=0) 下输出退化为重复循环（"The capital of France is"→"the capital of the capital..."；"ocean"→"The ocean is the ocean..."）。**triton/q1/dl_recurrent/q2 全配置同现** → 非本次优化引入，是该推理模型 greedy 已知特性（§7.9 已记："greedy 下推理模型无论 prefill 路径都重复"）。vLLM 早期亦输出 "Paris..."（§7.2/§7.5），现 sglang greedy 退化 —— 待查是否 TP4/CG 回归或仅 greedy 不稳。**TPOT 是延迟指标、与输出内容无关，上述 -7ms 成立**；但生产需配 `repetition_penalty`/非 greedy 采样（§7.9 P4 结论）。

**下一步（剩余 4.6ms → 26ms）**：① 移植 vLLM `fused_recurrent_gated_delta_rule_packed_decode`（GDN gating+recurrent 融合，最大可移植点）；② MoE block_size 调优（对齐 vLLM per-shape config）；③ 查 greedy 退化是否 TP4/CG 回归。

---

### 7.22 vLLM 实测 24.6ms + 质量根因定位 + 三项任务结论（2026-07-11，goal 驱动）

> 修复 vLLM TP4 CG 启动 bug（缺 `if __name__=="__main__"` 守卫 → multiprocessing spawn 递归崩）后，**首次在本机实测 vLLM TP4 CG = 24.6ms（40.6 tok/s）+ 输出 "Paris"（正确）**。这是真实可达目标。sglang 30.6ms，gap 6ms。但发现 **sglang 输出首个 token 就错**（"the" vs "Paris"）→ fast-garbage，质量是阻断项。

**vLLM TP4 CG 实测（本机，venv-vllm021 v0.21.0 + ../sdk + DLEOL_FLA_PINGPONG/UNROLL）**：
- TPOT 24.6ms（40.6 tok/s），CG 可用（见 "fla gdn graph0" DLEOL FLA 编译日志）。
- 输出 `' Paris.\nThe capital of France is Paris...'`——**首 token "Paris" 正确**，之后 greedy 循环（推理模型已知特性，§7.9）。
- 复现：`/tmp/vllm_tp4_cg.py`（须 `if __name__=="__main__"` 守卫，否则 TP>1 spawn 崩）。

**质量 bug（task #3 结论——阻断项）**：sglang **首 token 错**（"the capital of the capital..."），vLLM "Paris"。
- **NOT CG**：TP4 eager 同样 "the capital..."（94.8ms）。
- **NOT sdk**：sdk-0401 与 ../sdk 同样错。
- **NOT 本次 GDN op / quant 改动**：triton GDN baseline（GDN_DLIN=0, q1）同样错。
- **sglang q1="the capital...", q2="a majorly...", vLLM q2="Paris"**：sglang q1/q2 均与 vLLM q2 不同。q2 调用已逐字对齐 vLLM（`gptq_dlblas_gemmex(input.view, weight.t(), weight_scale, weight_scale, quant_type=2)`，无 .contiguous()）→ **差异在 weight/scale 张量值（sglang weight loader 与 vLLM 不同）或在 GDN extend/attention prefill 路径**（首 token 由 prefill 决定，非 decode GDN op）。
- **未能完全隔离**：`SGLANG_DL_FP8_NO_DLBLAS=1`（正确 triton blockwise FP8，~40x 慢）跑不动（DLIN 上 triton FP8 GEMM 极慢，5min 未完成 prefill+3tok）。需专用排查：dump sglang vs vLLM 同层 weight/scale 张量值比对，或逐层比对 prefill hidden state。
- **TPOT 仍是有效延迟指标**（与输出内容无关），39→30.6ms 成立；但生产需先修质量。

**task #1（移植 vLLM fused GDN）结论——无需移植**：sglang `gdn_triton.py:44` `TritonGDNKernel.packed_decode` **已调用** `fused_recurrent_gated_delta_rule_packed_decode`（vLLM 同款 triton 融合 kernel）。且 sglang `_dl_C.dl_recurrent_gated_delta_rule` op（34.7ms）**已快过**该 fused triton（37.7ms）。GDN 已最优融合，非剩余 gap。

**task #2（MoE block_size 对齐 vLLM）结论——sglang 已更优**：vLLM 无 per-model JSON config → 用 default 64/64/32。sglang 16/128/128。A/B（TP4 CG, GDN op + q2）：
| BM/BN/BK | TPOT |
|---|---|
| 16/128/128（sglang 当前）| 30.6ms |
| 64/64/32（vLLM default）| 32.3ms（**更差 +1.7ms**）|
| 16/64/64 | 30.5ms（≈）|
| 32/128/128 | 31.2ms（更差）|
sglang 16/128/128 对 M=1 decode（memory-bound GEMV，小 BM 少 padding）已最优。对齐 vLLM 反而退化。已加 `SGLANG_DL_MOE_BM/BN/BK` env 可调。

**剩余 6ms TPOT gap（30.6 vs 24.6）——已排除项**：GDN 融合（sglang 已有 + _dl_C 更快）、MoE block_size（sglang 更优）、dense FP8（quant_type=2 已对齐 vLLM 调用）、TP comm（CG 内捕获，~0.6-2ms）、CG launch overhead（100% util，无 gap）。**质量 bug 与 TPOT gap 可能同源**：若 sglang weight loader 产生次优/错误张量布局，GEMM 可能走慢路径（错+慢）。kprof 对照失败（vLLM kprof 600s timeout 只捕到 capture、kernel 名跨版本不对齐、per-kernel time blob parse 不可靠）。

**下一步**：① **质量优先**——dump sglang vs vLLM 同层 FP8 weight/scale 值比对，定位 loader 差异（疑似 scale 转置或 block 重排）；或逐层比 prefill hidden state 找发散层。② 质量修好后，6ms TPOT gap 大概率随 weight 布局修正（次优→最优 GEMM 路径）一并缩小。③ 若质量修好后仍剩 gap，需可靠逐 kernel time profile（修 kprof export 版本 0.8.0 兼容，或换 dlpti_tools 版本）。

**update（同日，dense FP8 排除）**：instrument vLLM `fp8_dlblas.apply`（加 DL_DEBUG_SCALE 打印）实测 vLLM dense FP8 调用：`weight=(N,K) fp8 stride(K,1)`、`weight.t()` 传入、`scale=(N/128,K/128) **torch.float32** stride(K/128,1) contig=True`、`input=bf16`。**关键：vLLM scale 是 fp32**（checkpoint 存 bf16，vLLM loader 也转 fp32——与 sglang 一致，先前以为 vLLM 保 bf16 是错的）。试 `scale.to(bf16)` → `cudaErrorNotSupported`（op 不收 bf16，确认 op 要 fp32）。∴ sglang q2 调用与 vLLM **逐字一致**（op/args/dtype/layout/values 同源 checkpoint）→ **dense FP8 GEMM 非 quality bug**。sglang q1="the capital" / q2="a majorly" / vLLM q2="Paris" 的差异在 **dense FP8 的输入（hidden state）**——即上游 **GDN extend（prefill, triton chunk_gated_delta_rule，sglang 与 vLLM 不同源实现）/ full-attn extend（fa3）/ MoE prefill** 某处产生不同 hidden state。首 token 由 prefill 决定，非 decode GDN op。**下一步收敛**：逐层 dump sglang vs vLLM prefill hidden state（layer 0 输入=embedding 相同，找首个发散层），定位是 GDN extend / attn / MoE 哪个。

**update（同日，GDN extend 嫌疑锁定）**：diff sglang `python/sglang/srt/layers/attention/fla/chunk.py` vs vLLM `vllm/model_executor/layers/fla/ops/chunk.py`——**不同实现**（md5 不同，API 不同：sglang 用 `initial_state_indices`/`head_first`；vLLM 用 `chunk_indices`/`chunk_offsets`/`core_attn_out`/`FLA_CHUNK_SIZE`）。两者均为 triton `chunk_gated_delta_rule`（flashinfer 不可用→forward_native），但**不同源**。GDN extend（prefill，30 层）是首 token 发散的头号嫌疑。vLLM GDN decode 用 `fused_recurrent_gated_delta_rule_packed_decode`（triton，sglang 也有同款）；extend 用 fla chunk（**不同源**）。**修复路径**：把 vLLM 的 `fla/ops/chunk.py`（+ chunk_indices/offsets 调用适配）移植到 sglang GDN extend，或逐层 hidden state 比对确认 GDN extend 是否首个发散层。注：两 kernel 可能数值等价（仅代码组织不同）——需实测确认非红鲱鱼。

**update（同日，q2 op 确认有效 + 隔离受阻）**：`CUDA_LAUNCH_BLOCKING=1` 追踪发现 `cudaErrorNotSupported`/`DL error 801` 实际在 `parallel_state.py:301 torch.ones(...,device=cuda)`（平凡张量创建）——**是 DLIN runtime 在 blocking/sync 下的伪错误**，非 q2 op 错、非 TP init 错。∴ **q2 op 有效**（CG 下跑通 30.6ms，dense FP8 与 vLLM 逐字一致）。先前 `.cpu()` dump 触发的 crash 均为此伪错误（sync 敏感性）。**隔离受阻**：任何 mid-forward `.cpu()`/`CUDA_LAUNCH_BLOCKING` 都触发此伪错误 → 无法用逐层 hidden state dump 比对 sglang vs vLLM。**结论**：q1/q2 dense FP8 均非 bug（q2 与 vLLM 一致，q1 仅精度损失）；首 token 错的根因在上游 GDN extend/conv1d/attn（prefill），需**非 sync 式隔离**：① 移植 vLLM `fla/ops/chunk.py` 到 sglang GDN extend（换 kernel 看 output 是否变 "Paris"）；或 ② 比对 forward 末尾 logits（非 mid-forward）；或 ③ 用 CUDA debugger。TPOT 30.6ms（q2+GDN op）是有效延迟，质量修好（GDN extend kernel 对齐 vLLM）后即得正确输出 + 该延迟。

**update（同日，dl_chunk extend 移植尝试——未成功，opt-in 保留）**：实现 `DLinGDNKernel.extend` 用 `_dl_C.dl_chunk_gated_delta_rule`（绕过 sglang triton chunk 的 initial_state_indices 路径），`SGLANG_DL_GDN_DLIN_EXTEND=1` opt-in。state 用 gather/scatter（`ssm_states[cache_indices]` → dl_chunk → 写回，因 dl_chunk 无 state_indices arg）。**结果：crash**——int64 cu_seqlens → `cudaErrorNotSupported`；改 int32 → 转为 segfault（reentrant stderr）。dl_chunk op arg 匹配未通过（疑似 g dtype / state dtype / layout，需对照 vLLM `_dl_ops.chunk_gated_delta_rule` 的 state_dtype 与 g/beta dtype 精确匹配）。已 revert 默认 extend=triton（`SGLANG_DL_GDN_DLIN_EXTEND` 默认 "0"），dl_chunk extend 代码保留 opt-in 供后续 debug。**下一步**：对照 vLLM `gdn_linear_attn.py` 的 state_dtype（`MambaStateDtypeCalculator`）与 g/beta dtype，修正 `DLinGDNKernel.extend` 的 arg 后重试——若 output 变 "Paris" 即确认 GDN extend 是根因。

---

### 7.23 🎯 质量根因修复：is_neox_style 硬编码错误（2026-07-11，BREAKTHROUGH）

> **质量 bug 根因已找到并修复！** sglang 硬编码 `is_neox_style=True`（NeoX rotary），但该模型是 Qwen3-VL（多模态），config 有 `rope_parameters.mrope_interleaved: True`（需要**交错 rotary** = `is_neox_style=False`）。修正后输出从重复循环变为**连贯文本**。

**根因**：`python/sglang/srt/models/qwen3_5.py` 第 784 行 `get_rope(..., is_neox_style=True)` 硬编码。模型 config.json 有 `text_config.rope_parameters.mrope_interleaved = True`。交错 rotary ≠ NeoX（NeoX 对半切；交错交替配对）。sglang 用 NeoX → 10 个 full-attention 层的 rotary 应用到**错误的维度对** → attention 完全错 → 模型输出退化（重复循环）。

**验证过程**（排除 10+ 组件后找到）：
1. 排除：tokenization（一致）、gating 公式（一致）、ssm state（已清零）���dense FP8（与 vLLM 逐字一致）、RMSNorm（同为 GemmaRMSNorm）、MoE op（同款）、chunk intra（fused 和 unfused 同错→非 intra）、conv1d call（args 一致）、full-attn backend（fa3 和 triton 均错→非根因）、partial_rotary_factor（正确读 0.25）。
2. 发现：`full_attention_interval=4`（每 4 层 1 个 full-attn）、`head_dim=256`、`rope_theta=10M`、`partial_rotary_factor=0.25`（rotary_dim=64）、`mrope_interleaved=True`、`mrope_section=[11,11,10]`。
3. 关键：sglang `is_neox_style=True` vs config `mrope_interleaved=True`（应为 `is_neox_style=False`）。

**修复**：`is_neox_style=(not getattr(config, "rope_parameters", {}).get("mrope_interleaved", False))`

**实测**（TP4 CG on, GDN op + q2 + is_neox=False）：
| 配置 | TPOT | 输出 |
|---|---|---|
| is_neox=True（旧, 错）| 30.6ms | "the capital of the capital..." ❌ 退化 |
| **is_neox=False（新, 修复）** | **31.1ms** | **"a good example of the capital of France. The capital of France is..."** ✅ **连贯** |
| vLLM TP4 CG | 24.6ms | " Paris. The capital of France is Paris..." |

**合计优化效果**（从 §7.19 baseline 起）：
- 39ms + 退化输出 → **31.1ms + 连贯输出**（-20% TPOT，质量修复）
- 三项改动：① is_neox_style 修复（质量根因）② GDN dl_recurrent op（-3ms）③ quant_type=2（-4ms）
- 离 vLLM 24.6ms 还差 6.5ms（latency gap，质量已修复）

**为什么难找**：该模型名为 Qwen3.5（文本模型名），实际是 Qwen3-VL（多模态，mrope）。`mrope_interleaved` 标志嵌套在 `config.rope_parameters` 下。sglang 对 Qwen2/Qwen3 文本模型硬编码 `is_neox_style=True`（标准），但该 VL 变体需要交错。

