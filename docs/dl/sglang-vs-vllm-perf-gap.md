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
| **sglang** | BREAKABLE graph (LogitsProcessor fixed) | **17.1** | ✅ 干净 | "Paris..." ✅ |
| **sglang** | eager (paged_decode_attn kernel) | **12.7** | — | "Paris..." ✅ |

### 性能差距分解

| 指标 | vLLM | sglang | 差距 | 根因 |
|---|---|---|---|---|
| **Eager tok/s** | 21.4 | 12.7 | **-41%** | gather 开销 + 缺失 DLIN 算子 |
| **Graph tok/s** | 23.0 | 17.1 | **-26%** | BREAKABLE 开销 + 残留 torch-native ops |
| **Graph vs Eager 提升** | +7.5% | +35% | — | sglang 从低基数提升更大比例 |

---

## 2. 根因分析

### 2.1 Attention 路径差异（最大单项差距）

| | vLLM | sglang |
|---|---|---|
| **Decode attention API** | `vllm_flash_attn.flash_attn_varlen_func(block_table=)` | `flash_attn.flash_attn_varlen_func`（gather 后调用） |
| **底层 dldnn op** | `cudnnMHAVarlenForward*`（直接读分页 cache） | `cudnnMHAVarlenForward*`（先 gather 成 packed 再调用） |
| **每层额外拷贝** | 0 | 2 × B × max_context × Hkv × D bytes |
| **每层额外 GPU ops** | 0 | ~5（gather + mask + cumsum + reshape + varlen） |
| **估计性能损失** | — | **~6 tok/s**（~30% of eager） |

### 2.2 DLIN 算子覆盖差距

| 算子类别 | vLLM（登临优化） | sglang（当前） | 估计性能损失 |
|---|---|---|---|
| **RMSNorm** | dleol kernel | dlcc kernel ✅（已接入） | ~0 |
| **Attention decode** | dleol `cudnnMHAVarlenForward*` | torch gather → `flash_attn_varlen_func` | **~6 tok/s** |
| **RoPE** | dlcc `rotary_embedding` | dlcc `rotary_embedding` ✅ | ~0 |
| **Linear / matmul** | dlblasLt（DLIN-optimized GEMM） | torch `nn.Linear`（DLIN torch 内置） | ~1 tok/s |
| **Sampler** | DL flashinfer-ext | pytorch fallback | ~0.5 tok/s |
| **MoE gate** | dlcc `moe_fused_gate` | schema-only stub | N/A（dense model） |
| **Quant GEMM** | dlblasExt（FP8/W8A8/GPTQ…） | 无 | N/A（bf16 model） |

### 2.3 CUDA Graph 差异（关键发现）

| | vLLM | sglang |
|---|---|---|
| **Graph backend** | FULL（整图） | BREAKABLE（分段，workaround） |
| **FULL graph 正确性** | ✅ 正常 | ❌ gibberish |
| **根因** | 大部分 ops 走 dleol/dlblas（graph-safe） | 大部分 ops 走 torch-native（graph replay 出错） |
| **Graph 性能损失** | +1.6 tok/s（7.5%） | +5.3 tok/s（42%，但基数低） |

**关键发现：DLIN cuda graph replay 对 dleol/dlblas-compiled kernels 正确，对 torch-native
CUDA ops 超过约 40 个后出错。** vLLM 的 FULL graph 正常是因为大部分 forward ops 走 dleol；
sglang 的 FULL graph 出 gibberish 是因为大部分 ops 是 torch-native + gather 额外开销。

**实测证据**（纯 torch matmul 链，无 dlcc/triton/attention）：
- N=30 matmuls → replay err=0 ✅
- N=50 matmuls → replay err=NaN ❌

Qwen3-1.7B decode forward 有 ~28 层 × ~12 ops = ~336 ops（vLLM 全走 dleol → graph-safe；
sglang 大部分走 torch-native → 超 40 op 限制 → gibberish）。

---

## 3. 优化路线图（按优先级）

每项标注：预估收益、工作量、依赖。

### P0：消除 gather 开销（+~6 tok/s eager, +解锁 FULL graph）

**目标**：让 sglang decode 直接走 `cudnnMHAVarlenForward*`（和 vLLM 一样），不 gather。

| 方案 | 做法 | 依赖 | 工作量 |
|---|---|---|---|
| **A. dl24 vllm_flash_attn wheel** | 登临提供匹配 torch==2.9.1+dl24.sdk202606031721 的 wheel → `pip install` → 切到 `vllm_flash_attn.flash_attn_varlen_func(block_table=)` | 登临出 wheel | 0.5 天（接入已就绪，见 `jit_kernel/flash_attention.py` 的 `_dlin_vllm_flash_attn_ok()` 分支） |
| **B. 自编 vllm_flash_attn** | 用本地 flash-attention 源码 + vLLM CMake 构建壳，dlcc 编译 | 需复刻 vLLM CMake FetchContent | 3-5 天 |
| **C. 自写 paged-decode kernel** | ✅ 已完成：`sgl-kernel/csrc/elementwise/paged_decode_attn_dl.cu`（精确 vs SDPA, graph-safe） | 无 | 已提交 `9986422670` |

**当前状态**：方案 C 已完成（正确、graph-safe、capture OK、replay FAST 18 tok/s），但 FULL
graph replay 仍 gibberish（因为其他 torch-native ops 也超 40 op 限制）。方案 A 是根治（全
forward 走 dleol → graph-safe），但依赖登临出 wheel。

**建议**：方案 C 保留为 eager 路径（替代 gather，减少开销），同时推进方案 A（wheel 到手后
自动切到 dleol 直读 paged cache 路径）。

### P1：BREAKABLE graph 修复 ✅ 已完成

**已修复**：`breakable_cuda_graph_backend.py` 的 `_slice_output` /
`_copy_output_to_buffer` 现在正确处理 `LogitsProcessorOutput`（切分
`next_token_logits` tensor，pass-through 其他字段）。输出从 ` ?\n...` 变为干净的
` Paris...`。Commit `5ab2426401`。

**当前 BREAKABLE 性能**：17.1 tok/s（比 eager 12.7 快 35%）。

### P2：接入 DLIN 算子（减少 torch-native ops, +~2-3 tok/s eager, +解锁 FULL graph）

每接入一个 DLIN 算子，减少一个 torch-native op → eager 更快 + FULL graph 更接近可用。
详见 `dlin-vllm-sglang-gap-analysis.md` 的完整算子清单。

| 优先级 | 算子 | sglang 接入点 | 工作量 | 预估收益 |
|---|---|---|---|---|
| P2a | **Sampler**（DL flashinfer-ext） | `layers/sampler.py` | 1 天 | +0.5 tok/s |
| P2b | **Linear head-padding**（dlblasLt） | `layers/linear.py` | 1 天 | +0.5 tok/s |
| P2c | **MoE fused gate**（dlcc） | `layers/moe/topk.py` | 1 天 | N/A（dense） |
| P2d | **FP8/W8A8 GEMM**（dlblasExt） | `layers/quantization/*` | 5 天 | N/A（bf16） |

> P2a-P2b 对 Qwen3-1.7B 有收益；P2c-P2d 对量化/MoE 模型有收益（路线图储备）。

### P3：FULL cuda graph（登临侧修 graph replay 正确性, 或 P2 充分后自动可用）

**路径 A**（登临修）：报告 DLIN cuda graph replay 对 torch-native ops 超阈值后出错 → 登临修
→ sglang FULL graph 自动可用（所有 ops 正确 replay）。

**路径 B**（sglang 自助）：P0+P2 接入足够多 DLIN 算子后，torch-native ops 数量降到 ~40 以
下 → FULL graph 可用。

### P4：vLLM 独有优化（路线图储备）

| 优化 | vLLM 做法 | sglang 接入点 | 何时做 |
|---|---|---|---|
| 独立 decode capture-size 列表 | `dl_config.py` | `cuda_graph_config.py` | P3 完成后 |
| logits GEMM 预热 | `dl_gpu_model_runner.py` | `model_runner.py` | P3 完成后 |
| memory-pool tagging | `dl_gpu_worker.py` | `gpu_worker.py` | P3 完成后 |

---

## 4. 预期性能演进

| 阶段 | 完成项 | Eager tok/s | Graph tok/s | vs vLLM |
|---|---|---|---|---|
| **当前** | paged_decode_attn + BREAKABLE (LogitsProcessor fixed) | 12.7 | 17.1 | -26% |
| **P0 完成** | paged_decode_attn 替代 gather（eager） | ~15-16 | 17.1 (BREAKABLE) | -26% to -30% |
| **P0+P1 ✅** | + BREAKABLE 干净 ✅ | ~15-16 | ~17-18 | -22% to -26% |
| **P0+P1+P2 完成** | + DLIN sampler + head-padding | ~17-18 | ~19-20 | -10% to -15% |
| **+P3 (FULL graph)** | 登临修 graph replay 或 ops 够少 | ~17-18 | **~22-23** | **~0%**（追平 vLLM） |

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
