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

### 2.4 Batch-scaling：真正的差距在高 batch compute slope（第一性原理裁决）

> torch.profiler 在 DLIN 崩溃，故用 batch-scaling 分解 `step_time = overhead + per_seq·bs`。

eager 模式实测（Qwen3-1.7B，DLIN KS38，NEW=64）：

| bs | vLLM step_ms | sglang step_ms | sglang/vLLM |
|---|---|---|---|
| 1 | 46.7 | 48.2 | 1.03×（parity） |
| 4 | 51.3 | 59.5 | 1.16× |
| 16 | 54.7 | 77.3 | 1.41× |
| 64 | 67.6 | 147.2 | **2.18×** |

线性拟合：vLLM `step ≈ 46.4 + 0.33·bs`；sglang `step ≈ 46.6 + 1.57·bs`。

- **截距相同（~46.5ms overhead）**：batch=1 双方等价；engine round-trip overhead 非差距来源。
- **sglang per-seq 斜率 1.57ms = vLLM 0.33ms 的 4.75×**：真差距，**只在高 batch 显现**
  （bs=64：vLLM 947 vs sglang 435 tok/s）。
- **matmul 微基准 + FLOPs 核算**（bs=64 decode ≈180 GFLOP/step）：

  | | in-model TFLOP/s | vs isolated `torch.matmul`(4.8) |
  |---|---|---|
  | isolated `torch.matmul` [64,2048]×[2048,6144] | 4.8 | 1.0× |
  | **sglang eager（in-model）** | ~1.8 | **0.37×（2.7× 损耗）** |
  | **vLLM eager（in-model）** | ~8.5 | 1.77×（**超过 isolated**） |

  sglang 把 matmul 跑在 isolated 速率的 0.37× —— **未融合的小 matmul**（28层×~5个，shape 小 → 低
  TFLOP/s）+ launch 开销；vLLM 靠 **fusion（QKV/gate_up 融合→更大 matmul）+ dlblasLt** 反超 isolated。
  二者相乘 ≈ 4.75× 斜率。

- **cuda graph 在 DLIN 上 net-negative**（实测 sglang graph 比 eager **慢**，全部 batch：bs=1 60.5 vs
  48.2ms，bs=64 312.6 vs 147.2ms）。graph replay/static-pool 开销 > launch 节省，且随 batch 增长。
  ∴ **graph 不是 sglang 的解法**。

**裁决**：本报告早先聚焦的「batch=1 -9% eager 差距」是**测错对象**——batch=1 双方 overhead-bound
等价（差距 <3%，run 间方差）。真差距是**高 batch 的 per-seq compute 效率**（2.18× @ bs=64），根因
= **fusion 缺失（主）+ dlblasLt（次 ~1.9×）**，**不是** graph。复现：
`scripts/dl/bench_batch_scaling.py`（sglang eager/graph）、`bench_vllm_batch_scaling.py`（vLLM）。

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

### P2：fusion + dlblasLt（高 batch 差距的根治，~4.75× per-seq）⬆ 最高优先级

**§2.4 裁决**：真差距是高 batch 的 per-seq compute 效率（sglang 斜率 1.57ms = vLLM 0.33ms 的 4.75×）。
matmul 微基准证因：sglang in-model 仅 ~1.8 TFLOP/s（isolated `torch.matmul` 的 0.37×，因**未融合的
小 matmul** + launch 开销）；vLLM 靠 **fusion + dlblasLt** 达 ~8.5 TFLOP/s（超 isolated）。**graph 在
DLIN net-negative（非解法，见 §2.4）**。这取代原「batch=1 -9% eager」目标——P2 在 batch=1 无收益，
在 serving batch 是 **2-5× 吞吐**。

| 优先级 | 算子 | sglang 接入点 | 工作量 | 预估收益（高 batch） |
|---|---|---|---|---|
| **P2a** | **fused QKV / fused gate_up MLP**（小 matmul→大 matmul，提 TFLOP/s） | model layer / `layers/linear.py` | 2-3 天 | **主杠杆**（in-model 0.37×→接近 1×） |
| P2b | **Linear GEMM → dlblasLt** | `layers/linear.py` | 2-3 天 | 次（~1.9× raw GEMM） |
| P2c | **Sampler**（DL flashinfer-ext） | `layers/sampler.py` | 1 天 | 小（batch=1 overhead 侧） |
| P2d | **MoE gate / FP8-W8A8 GEMM**（dlblasExt） | `layers/{moe,quantization}/*` | 5 天 | N/A（dense/bf16，储备） |

> P2a（fusion）杠杆 > P2b（dlblasLt）：sglang 的 2.7× in-model 损耗主要来自未融合小 matmul，融合后
> TFLOP/s 才逼近 isolated，dlblasLt 再叠加 ~1.9×。详见 `dlin-vllm-sglang-gap-analysis.md`。

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

> **指标重定向（§2.4）**：batch=1 双方 overhead-bound 等价（~47ms），无差距可追。真正指标是
> **高 batch 吞吐**（bs≥16），差距在那里（sglang 435 vs vLLM 947 tok/s @ bs=64）。

| 阶段 | 完成项 | bs=1（parity） | bs=64 tok/s | vs vLLM @bs=64 |
|---|---|---|---|---|
| **当前** | paged_decode_attn FULL（干净，无需 vllm_flash_attn） | ~20（等价） | 435 | **-54%**（2.18× 慢） |
| **+P2a（dlblasLt GEMM）** | Linear 走 dlblasLt | ~20 | ~700-900 | -5% to -25% |
| **+P2b（fused QKV/MLP）** | 进一步降 kernel 数/提效 | ~20 | **~900+** | **~0%**（追平 vLLM） |

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
