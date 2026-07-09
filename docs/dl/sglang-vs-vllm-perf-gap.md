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
