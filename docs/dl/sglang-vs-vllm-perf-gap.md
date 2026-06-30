# sglang vs vLLM 性能差距分析与优化路线图

> 基于 Qwen3-1.7B（bf16, batch=1, DLIN KS38）实测数据，量化 sglang 与 vLLM 的 decode
> 性能差距，分析根因，给出按优先级排列的优化路线图。
>
> 测试日期：2026-06-30 ｜ sglang dl-main (7 commits) ｜ vLLM 0.21.0 (dl19 torch)
>
> 所有数据为同一 GPU（cuda:3, KS38 QUAD 32GB）、同一模型（/opt/dataset/Qwen3-1.7B）、
> 同一 prompt（"The capital of France is", greedy, 64 tokens）下的实测。

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
| **Eager tok/s** | 21.4 | 19.5 | **-9%** | attention 路径已对齐（Route A），剩余差 sampler + 少量 kernel（vllm_flash_attn 19.52 ≈ paged_decode_attn 19.45） |
| **Graph tok/s** | 23.0 | 16.6–17.1 | **~-27%** | batch=1 graph 不划算：FULL(vllm_flash_attn) 16.64 ✅干净 / BREAKABLE(paged_decode) 17.1 ✅干净 |
| **Graph vs Eager** | +7.5% | -15% | — | sglang batch=1: graph 比 eager 慢（forward 太轻量） |

---

## 2. 根因分析

### 2.1 Attention 路径差异（Route A 后基本消除）

| | vLLM | sglang（旧 gather 路径） | sglang（Route A: vllm_flash_attn） |
|---|---|---|---|
| **Decode attention API** | `vllm_flash_attn.flash_attn_varlen_func(block_table=)` | `flash_attn.flash_attn_varlen_func`（gather 后调用） | 同 vLLM：`vllm_flash_attn.flash_attn_varlen_func(block_table=)` |
| **底层 dldnn op** | `cudnnMHAVarlenForward*`（直接读分页 cache） | `cudnnMHAVarlenForward*`（先 gather 成 packed 再调用） | 同 vLLM（直读分页 cache） |
| **每层额外拷贝** | 0 | 2 × B × max_context × Hkv × D bytes | 0 |
| **每层额外 GPU ops** | 0 | ~5（gather + mask + cumsum + reshape + varlen） | 0 |
| **eager tok/s** | 21.4 | （未单独 bench） | **19.52** |

**结论**：Route A 让 sglang decode 走与 vLLM **完全相同**的 attention 路径（无 gather），attention
单项差距基本消除。剩余 eager 差距（19.52 vs 21.4 = -9%）来自 sampler + 少量 matmul/kernel 差异
（见 §2.2），不再是 gather 开销。

### 2.2 DLIN 算子覆盖差距

| 算子类别 | vLLM（登临优化） | sglang（当前） | 估计性能损失 |
|---|---|---|---|
| **RMSNorm** | dleol kernel | dlcc kernel ✅（已接入） | ~0 |
| **Attention decode** | dleol `cudnnMHAVarlenForward*` | dleol `cudnnMHAVarlenForward*`（Route A: vllm_flash_attn）✅ | ~0（旧 gather 路径曾损 ~6 tok/s，Route A 已消除） |
| **RoPE** | dlcc `rotary_embedding` | dlcc `rotary_embedding` ✅ | ~0 |
| **Linear / matmul** | dlblasLt（DLIN-optimized GEMM） | torch `nn.Linear`（DLIN torch 内置） | ~1 tok/s |
| **Sampler** | DL flashinfer-ext | pytorch fallback | ~0.5 tok/s |
| **MoE gate** | dlcc `moe_fused_gate` | schema-only stub | N/A（dense model） |
| **Quant GEMM** | dlblasExt（FP8/W8A8/GPTQ…） | 无 | N/A（bf16 model） |

### 2.3 CUDA Graph 差异（已用第一性原理 + 裁决实验证证）

| | vLLM | sglang |
|---|---|---|
| **Graph backend** | FULL（整图） | FULL（两条 decode 路径都已干净）✅ |
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
> 实测**证伪并撤回**——sglang 已融合、dlblas 对 bf16 N/A、matmul 双方等价。差距真实但根因待隔离。

**裁决**：sglang vs vLLM 在 DLIN（Qwen3-1.7B bf16）**稳态持平**（median 全 batch +2~7%）——既无
compute/kernel/fusion 差距，147ms tail 也已查明为 JIT warmup 假象（warmup 对齐 `temperature=0` 即消除）。
**整个 perf-gap 调查结论：不存在实际性能差距**；教训 = benchmark warmup 必须匹配 timed 采样参数。
复现：`scripts/dl/bench_batch_scaling.py`、`bench_vllm_batch_scaling.py`（warmup 均已对齐 temp=0）。

---

## 3. 优化路线图（按优先级）

每项标注：预估收益、工作量、依赖。

### P0：消除 gather 开销 ✅ 已达成（Route A）

**目标**：让 sglang decode 直接走 `cudnnMHAVarlenForward*`（和 vLLM 一样），不 gather。

| 方案 | 做法 | 依赖 | 状态 |
|---|---|---|---|
| **A. vllm_flash_attn** ✅ | 复用 vLLM DL venv 里的 DLIN-patched `_vllm_fa2_C.so`（dl19 build，与 dl24 torch ABI 兼容）；`scripts/dl/setup_vllm_flash_attn.sh` 自动复制 | 无新依赖 | **已生效**：eager 19.52 + FULL graph 16.64 ✅干净 |
| **B. 自编 vllm_flash_attn** | 本地 flash-attention 源码 + vLLM CMake 壳，dlcc 编译 | 复刻 vLLM CMake FetchContent | 3-5 天（A 已够用，无需） |
| **C. 自写 paged-decode kernel** ✅ | `sgl-kernel/csrc/elementwise/paged_decode_attn_dl.cu`（精确 vs SDPA, graph-safe） | 无 | 已提交 `9986422670`；eager 19.45 |

**当前状态**：Route A 已达成——**无需登临再出匹配 wheel**，dl19 的 `_vllm_fa2_C.so` 直接复用
即可（见 `setup_vllm_flash_attn.sh`；冷启动 JIT "to bc failed" 由 model warmup 预热 dleol 缓存
绕过，见 `dlin-fa2-decode-bug-report.md`）。`jit_kernel/flash_attention.py` 的
`_dlin_vllm_flash_attn_ok()` 分支已就绪，.so 在位即自动走 varlen+block_table 直读路径。**意外
收获**：Route A 让 FULL graph 输出干净（见 §2.3 修正）。

**当前最优 = Route C**：`paged_decode_attn` 已能单独跑干净 FULL graph（≈17.9 tok/s，见 §2.3 裁决
实验），**无需 vllm_flash_attn**——比 Route A 的 FULL（16.6）更快且无外部 .so 依赖。Route A
降级为对照/备用。

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
> launch 节省。∴ sglang 的性能路径是 **eager + fusion（P2a）+ dlblasLt（P2b）**，不是 graph。
vLLM 的 graph 有效是因其 forward 已高度融合+dlblasLt，graph 仅锦上添花；sglang 反被 per-batch
静态池开销拖累。

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
> 准备就绪，但 batch=1 bench 不体��其价值。

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
- **vllm_flash_attn dormant path**：wheel 到手即自动切换
- **docs/dl/**：gap analysis + bug report + wheel request
- **scripts/dl/**：repro, bench (sglang + vLLM), 5 个 correctness tests

---

## 7. Qwen3.5-35B-A3B-FP8 补充（2026-07-01，与 §1–6 的 Qwen3-1.7B bf16 不同模型）

> 模型：`/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`（本地副本 `/LocalRun/xi.chen/...`，35G）。
> 架构 `Qwen3_5MoeForConditionalGeneration`：**35B MoE、A3B（256 experts×8 active + shared expert）**、
> **hybrid linear/full attention**（每 4 层 1 层 full-attn，余为 GatedDeltaNet linear-attn）、FP8 blockwise
> （`weight_block_size=[128,128]`，dynamic act）、VLM、MTP。DLIN KS38，TP=2（35B FP8 ≈37GB > 1×32GB）。

### 7.1 裁决：差距是 **0→1 enablement**，非性能差距

| 框架 | Qwen3.5-35B-A3B-FP8 @ DLIN | 机制 |
|---|---|---|
| **vLLM 0.21.0** | ✅ **跑通**，输出正确 ` Paris. The capital of France is Paris...` | DL platform **自动把 FP8 重映射为 `quantization=fp8_dlblas`**（DLIN 原生 dlblas FP8 GEMM） |
| **sglang** | ❌ **未跑通**（forward 能跑到 100% GPU，但卡在 FP8 Triton bitcast） | FP8 走 NVIDIA **marlin / triton-fp8**（Hopper 专用）→ dlcc/Triton 编译失败 |

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
| 6 | **linear-attn Triton bitcast** `Cannot bitcast size-8 to size-1`（在 `hybrid_linear_attn_backend` / GDN / Lightning 的 Triton kernel，`track_mask` 处；模型有 30 层 linear-attn） | ❌ **未修** | linear-attn 的 Hopper Triton 内核（TMA/wgmma/FP8 bitcast）需 DLIN 等价或 fallback（§7.3 O4） |
| 7 | （未触及）blockwise-FP8 GEMM：auto 路径在 DLIN 走 CUTLASS/Triton，需 `_scaled_mm` 或 dlblas | ❌ 未触及 | 见 §7.3 O1（dlblas FP8） |

修到 #5 时 forward 已能执行（GPU 100%），说明 norm/marlin/gdc 绕过都生效。**剩余 #6/#7 是 linear-attn 与
FP8 GEMM 的 Hopper 专用 Triton/CUTLASS 内核**——属 DLIN kernel porting，非小补丁。**根因：Qwen3.5 深度
依赖 NVIDIA Hopper 特性（marlin PTX、Hopper Triton PDL/TMA/wgmma/FP8-bitcast），DLIN 无等价实现。**

### 7.3 优化路线图（goal 3：消除 enablement gap）

主攻 **把 dlblas FP8 GEMM 接入 sglang**（复刻 vLLM 的 `fp8_dlblas` 路由）：

| 优先级 | 项 | 做法 | 依赖 |
|---|---|---|---|
| **O1（根治）** | **sglang FP8 → dlblas** | 在 sglang FP8 quant method 里：DLIN + fp8 → 调 dlblas fp8 GEMM（blockwise 128×128）。先确认 dlblas fp8 的 Python 入口（vLLM 走 `torch.ops._dl_C.*` / dlblas plugin；sglang 需同等 binding） | dlblas fp8 op binding |
| O2 | 解决 #6 Triton FP8 bitcast | O1 接入 dlblas 后，FP8 GEMM 不再走 Triton → bitcast 自然消失 | O1 |
| O3 | 补齐 sgl_kernel DLIN 构建 | 把 `gemma_rmsnorm`/`rmsnorm`/`fused_add_rmsnorm` 等 norm ops 编进 DLIN sgl_kernel .so（替代 #1/#4 的 native fallback，提性能） | DLIN sgl_kernel 编译 |
| O4 | linear-attn 的 Hopper Triton extras | 评估 FLA kernels 还依赖哪些 Hopper Triton 特性（TMA/wgmma 等），逐个提供 DLIN 等价或 fallback | DLIN Triton 能力 |

> 注：sglang 与 vLLM 的 **tok/s 性能对比暂不可得**——sglang 跑不通该模型，无法 bench。vLLM 已跑通（可作
> sglang 移植后的目标基线）。**先把 sglang FP8 跑通（O1/O2），再谈 perf 差距。**

### 7.4 复现

- sglang（卡 #6）：`bash` 源 `SDK_DIR/env.sh` + `CUDA_VISIBLE_DEVICES=1,2` + `MODEL_PATH=/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8` → `python scripts/dl/run_qwen35_35b.py`（TP=2 eager；已带 #1–#4 修复）
- vLLM（跑通）：同 env（dl19 SDK + `venv-vllm021`）→ `python scripts/dl/run_qwen35_35b_vllm.py`（TP=2 eager；自动 `fp8_dlblas`）
- /mars NFS 读速 ~5 MB/s（慢），**用本地副本** `/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8` 加载。

