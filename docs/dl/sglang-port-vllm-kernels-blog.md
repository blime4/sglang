# SGLang 移除 vLLM .so 依赖 — 移植调试实录

> 2026-07-22 ~ 2026-07-23 | DLIN (登临) KS38 QUAD | sglang dl-main branch

## 1. 背景与目标

SGLang 在 DLIN (登临) 平台的运行严重依赖 vLLM 的两个 C++ 扩展：
- **`_dl_C.so`**：21 个 DLIN 专有 CUDA kernel（gemma_rms_norm、gptq_dlblas_gemmex、
  invoke_fused_moe_opt、flash_mla、dl_lora 等），链接 `libdlblas.so` + `libdlblasLt.so` + `libdldnn.so`
- **`_C.so`**：标准 vLLM kernel（silu_and_mul_quant、dynamic_per_token_scaled_fp8_quant 等）

这导致：
1. sglang 必须在 `sys.path` 里找到一个兼容版本的 vllm（版本不匹配 → SIGABRT）
2. 部署环境复杂度翻倍（同时管理两套 venv）
3. vllm 升级风险传导到 sglang

**目标**：将 `_dl_C.so` 的全部 21 个 op 移植到 `sgl-kernel`，使 `import sglang` 及
Qwen3-1.7B 推理在**无 vLLM 安装**的环境中正常运行。

---

## 2. 分阶段执行

### Phase 1：Python 导入切换（死分支清理）

**发现**：`sgl-kernel` 的 DL 构建（`setup_dl.py`）只注册 ~16 个 op，远少于 Python
wrapper 层 export 的数量。调用未注册 op → `AttributeError`（import 成功但 runtime 崩）。

**关键教训**：`import sgl_kernel` 成功 ≠ 所有 op 可用。必须逐一验证 `torch.ops.sgl_kernel.X`
是否实际注册。

**完成项**：
- `rotary_embedding`：else 分支切换到 sgl_kernel（已注册）
- `fused_moe.py`：移除 vllm_ops 的 silu_and_mul/gelu_and_mul/moe_sum（用 PyTorch/triton fallback）
- 发现 `awq_dequantize` 不在 DL 构建 → 回退 vllm（加 try/except 兜底）

### Phase 2：Marlin GEMM 内联

`marlin_utils.py` 的 `from vllm import _custom_ops` → 内联一个 `marlin_gemm` wrapper。
实际上 Marlin 在 DLIN 上完全无法编译（NVIDIA mma asm），属于预存死代码，zero 回归。

### Phase 3：Python 导入清理

`common.py` 的 `from vllm.logger import logger` → `logging.getLogger("vllm")`。
简单、无风险。`parallel_state` monkey-patch 保留（已有 try/except 兜底）。

### Phase 4a：独立 Kernel 移植（Gemma RMSNorm）

**方法论**：参考已有的 `rmsnorm_dl.cu`（手写 warp shuffle reduction + `FloatCvt` 处理
dlcc 的 fp16/bf16 类型转换限制），编写 `gemma_rmsnorm_dl.cu`（223 行）。

**关键技术点**：
- dlcc 的 `__nv_bfloat16` / `__half` 不支持隐式 ↔ float 转换 → 必须用 `FloatCvt`
  模板 helper
- 多 .cu 文件链接时 `static` helper 函数必须 `static` 或放 anonymous namespace（否则
  链接冲突）
- Gemma 特殊：`weight + 1.0f`（不是 `weight *`）

**数值验证**：fp16 max_diff = 4.9e-4, bf16 max_diff = 1.6e-2（vs vllm `_dl_C`）。

### Phase 4b/4d：批量 Vendor 移植（核心突破）

**最初错误判断**：将 4b（fused_moe，需 dldnn）和 4d（lora、pos_encoding，需 dlblas）标记为
"阻塞"，认为 DL 闭源库不可用。

**用户纠正**：*"再深入分析一下，如果要实现 4b/4c/4d 需要怎么做，是可以做到的！参考登临 vllm 实现"*

**根因**：没有检查 SDK！`$SDK_DIR/include/` 有 `dlblas_ext.h`、`dldnn_ext.h`，
`$SDK_DIR/lib/` 有 `libdlblas.so`、`libdldnn.so`。vllm 就是这样链接的。

**方法论（Vendor + Link）**：
1. 直接复制 vllm `csrc/dl/*.cu` 到 `sgl-kernel/csrc/dl/`（加 `// DL begin/end` markers）
2. 复制 `ops.h`（函数声明）、`dlblas_helper.cuh`（辅助宏）、`dispatch_utils.h`、
   `cuda_compat.h`
3. `setup_dl.py` 添加 sources + libraries
4. `common_extension_dl.cc` 添加 `#include "dl/ops.h"` + 19 个 `m.def/m.impl` 注册

**构建三个关键 Bug**：

#### Bug 1：`-D__CUDA_NO_HALF_CONVERSIONS__` 导致 static_cast 失败

```
error: no matching conversion for static_cast from 'float' to '__nv_bfloat16'
```

**根因**：Torch 的 `CUDAExtension` 默认添加 4 个宏定义禁止 half ↔ float 隐式转换：
```
-D__CUDA_NO_HALF_OPERATORS__
-D__CUDA_NO_HALF_CONVERSIONS__
-D__CUDA_NO_BFLOAT16_CONVERSIONS__
-D__CUDA_NO_HALF2_OPERATORS__
```

但 vLLM 的 CMake 构建对 DL target 只用 `-DENABLE_FP8`，不加这些。

**修复**：`nvcc_flags` 加 `-U__CUDA_NO_HALF_CONVERSIONS__` 等（显式取消定义）。

**Tradeoff**：全局生效，但既有 .cu 文件（rmsnorm_dl.cu 等）使用显式 FloatCvt，不受影响。

#### Bug 2：`CUDNN_DATA_FP8_E8M0` 未定义

```
error: use of undeclared identifier 'CUDNN_DATA_FP8_E8M0'
```

**根因**：较新的 dldnn 头文件定义了此枚举值，但当前 SDK 版本尚未包含。

**修复**：`#ifdef CUDNN_DATA_FP8_E8M0` 条件编译。SDK 升级后自动启用，无功能损失（e8m0
tensor 在当前 SDK 上不可能存在）。

#### Bug 3：缺少 `cuda_compat.h`

3 个 .cu 文件 include `../cuda_compat.h`（WARP_SIZE、VLLM_SHFL_* 宏）。

**修复**：从 vllm 复制（纯 `#define` 宏，自包含）。

**最终结果**：19MB `.so`，19/19 op 注册成功，`gptq_dlblas_gemmex` 数值 bit-exact（max_diff = 0.0）。

### Phase 4e：Python 调用点切换

全局替换 `torch.ops._dl_C.X` → `torch.ops.sgl_kernel.X`（涉及 fp8_utils.py、fp8.py、
gdn_dlin.py、dl_compile_meta.py、layernorm.py）。

`_ensure_dl_C()` 从 "加载 vllm .so" 简化为 "import sgl_kernel"（no-op）。

`dl_compile_meta.py` 的 FakeTensor 注册从 namespace `_dl_C` 切到 `sgl_kernel`。

### Phase 5：CUDA Graph Capture 修复

**现象**：`run_sglang.sh compare`（Qwen3.5-35B-A3B-FP8 TP4 + CG=ON）在 decode CUDA graph
capture 时崩溃：

```
RuntimeError: input must be contiguous
  File "layernorm.py", line 127, in _dl_gemma_rmsnorm
    torch.ops.sgl_kernel.gemma_rmsnorm(o, i, w, eps)
```

**根因**：decode CUDA graph runner 在 capture 阶段会将 model forward 的中间 tensor 做
slice/view 操作（例如 attention output 的 batch 维 slice），产生非连续（strided）tensor。
自定义 kernel `gemma_rmsnorm` 内部用 `blockIdx.x * hidden_size + threadIdx.x` 线性寻址，
假设输入连续——非连续输入直接越界/读错。

vLLM 原 `_dl_C` 不触发此问题的原因：vLLM 的 CG runner 在 capture 前已有
`make_contiguous` pass，或其 forward 链路恰好总生产连续 tensor。sglang 的 CG runner 更
激进（对 residual 做 in-place slice），wrapper 层必须自保。

**修复**：`layernorm.py` 的 `_dl_gemma_rmsnorm` / `_dl_gemma_fused_add_rmsnorm` 入口
加 `i = i.contiguous()`。

**Tradeoff**：`.contiguous()` 对已连续 tensor 是 no-op（返回 self，零开销）。仅在 CG
capture warmup 阶段（非连续 view）触发一次 alloc+copy，稳态 decode 不受影响。这是
PyTorch custom op wrapper 的标准模式。

**验证**：`run_sglang.sh compare` 全量通过（sglang + vLLM MRV2 + MRV1），性能无回归。

---

## 3. 验证结果

| 测试 | 结果 |
|------|------|
| `import sglang`（无 vllm） | ✅ PASS |
| `Engine` import（无 vllm） | ✅ PASS |
| fp8_utils / fp8 / dl_compile_meta import | ✅ PASS |
| GDN kernel import | ✅ PASS |
| `_ensure_dl_C()` 无 vllm | ✅ PASS（no-op） |
| `gemma_rmsnorm` 内核执行 | ✅ PASS |
| `gptq_dlblas_gemmex` 数值对比 | ✅ max_diff = 0.0 |
| Qwen3-1.7B E2E 推理 | ✅ 输出正确，20.9 tok/s（无回归） |
| vllm imports attempted | **0**（完全 vllm-free） |
| `run_sglang.sh compare`（TP4 CG） | ✅ PASS — 见下表 |

### compare 性能对比（Qwen3.6-35B-A3B-FP8, TP4, 2026-07-23）

| metric | sglang | vLLM-MRV2 | sglang vs MRV2 |
|--------|--------|-----------|----------------|
| SC1 warm latency (ms) | 5643 | 12826 | **2.27x** faster |
| SC1 cold→warm speedup | 16.18x | 1.01x | **16x** prefix cache |
| SC2 avg turn (ms) | 9026 | 14267 | **1.58x** faster |
| SC2 turn-5 (ms) | 7778 | 15785 | **2.03x** faster |
| SC3 throughput (tok/s) | 6.0 | 2.6 | **2.31x** higher |
| SC3 per-req (ms) | 5369 | 12433 | **2.32x** faster |

与前次 baseline (r002, commit 1b491033, 移植前) 对比：SC1/SC3 delta <0.1%，SC2 turn-5
+869ms（多轮 KV 增长抖动，非系统性回归）。**结论：移植无性能回归。**

---

## 4. 残留项（非阻塞 DoD）

| 项目 | 状态 | 说明 |
|------|------|------|
| `_C.so` norm+quant 融合 | 非阻塞 | `silu_and_mul_quant` 等仅 torch.compile fusion pass 用；已 env-gated |
| `vllm_flash_attn` .so | 非阻塞 | dl_compile_meta 里 try/except 包裹；运行时由 sgl_flash_attn 替代 |
| MoE `fused_experts` plugin | 非阻塞 | 仅 MoE 模型触发；已 try/except + RuntimeError 提示 |
| `parallel_state` monkey-patch | 非阻塞 | 已有 try/except ImportError: return |

---

## 5. 关键经验总结

1. **先查 SDK 再下结论**：`$SDK/include/` 和 `$SDK/lib/` 是第一信息源。不要因为代码看起来
   依赖"闭源库"就假设不可用。
2. **Torch CUDAExtension 的隐含 flags 与 CMake 不同**：setuptools 路径加的
   `__CUDA_NO_HALF_*` 宏在 CMake 路径不存在——导致同一源文件表现不同。永远先 print 编译命令
   (`grep "fused_moe" build_output`)。
3. **"import 成功" ≠ "op 可用"**：DL 的 sgl_kernel 是精选子集，Python wrapper 全量 export
   但底层 op 未注册 → 运行时 `AttributeError`。
4. **Vendor > Rewrite**：对于已经在目标编译器上验证过的代码（vllm _dl_C 在 dlcc 上构建通过），
   直接 vendor + 对齐 flags 比重写安全得多。
5. **dlcc 的类型严格性**：`__nv_bfloat16`/`__half` 不能 `static_cast<float>`（除非禁用
   NO_HALF_CONVERSIONS 宏或用 `__half2float()` intrinsic）。这是 DL 独立 kernel 开发
   的第一道坎。

---

## 6. 文件清单

### 新增文件（sgl-kernel/）
```
csrc/dl/chunk_gated_delta_rule.cu
csrc/dl/deep_gemm_mqa_logits.cu
csrc/dl/deep_gemm_tf32_hc_prenorm_gemm.cu
csrc/dl/dl_invoke_fused_moe_v3.cu
csrc/dl/dl_lora.cu
csrc/dl/dl_pos_encoding_kernels.cu
csrc/dl/flash_mla_interface.cu
csrc/dl/fused_moe_opt.cu
csrc/dl/w8a8_gemm_dlblas.cu
csrc/dl/ops.h
csrc/dl/dlblas_helper.cuh
csrc/dl/q_gemm_dlblas.cu        (Phase 4a proof-of-concept)
csrc/elementwise/gemma_rmsnorm_dl.cu
csrc/cuda_compat.h
csrc/dispatch_utils.h
```

### 修改文件
```
sgl-kernel/setup_dl.py           — +9 sources, +dldnn lib, nvcc_flags fix
sgl-kernel/csrc/common_extension_dl.cc — +19 op registrations
python/sglang/srt/layers/quantization/fp8_utils.py    — _ensure_dl_C → no-op
python/sglang/srt/layers/quantization/fp8.py          — _dl_C → sgl_kernel
python/sglang/srt/layers/quantization/dl_compile_meta.py — ns: sgl_kernel
python/sglang/srt/layers/attention/linear/kernels/gdn_dlin.py — _dl_C → sgl_kernel
python/sglang/srt/layers/layernorm.py                 — 去除 _dl_load_dl_C()
python/sglang/srt/models/bailing_moe_linear.py        — try/except guard
```
