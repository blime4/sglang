# DLEOL JIT Bug 报告：sglang vs vLLM MoE 性能差距的阻塞点

**报告日期**: 2026-07-13
**SDK 版本**: `../sdk/env.sh` (dl19-matching)
**模型**: Qwen3.5-35B-A3B-FP8, TP=4

---

## 概述

sglang 在 Qwen3.5-35B TP4 decode 中的 GPU forward 时间为 **27.5ms**，vLLM 为 **18.3ms**（gap 9.2ms）。
两者调用**完全相同的 `_dl_C.so` 操作**（已验证），但 DLEOL JIT 为 sglang 编译了更慢的 kernel 变体。

**尝试切换到 vLLM 更快的 kernel 变体时，DLEOL JIT 崩溃**。本报告列出 3 类 JIT bug，每类含最小复现。

---

## Bug 1: `K % VEC_K == 0` static_assert（JIT 编译时）

### 症状
```
static_assert(K % VEC_K == 0, "K must divied by VEC_K");
```
JIT 编译 `invoke_fused_moe_opt` 的某些参数组合时触发。

### 触发条件
改变以下任一参数（相对于已成功编译的 baseline key）：
- `mul_routed_weight`: True → False
- 加载 vLLM `_C.cpython-312-x86_64-linux-gnu.so`（改变 JIT 全局状态）
- `try_get_optimal_moe_config` 返回不同的 BLOCK_SIZE_M/N/K

### 最小复现
```python
import torch
torch.ops.load_library("vllm/_dl_C.cpython-312-x86_64-linux-gnu.so")
_G = torch.ops._dl_C.invoke_fused_moe_opt

# 这些参数组合能成功编译（baseline key）
x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
w = torch.randn(256, 256, 2048, device="cuda").to(torch.float8_e4m3fn)
ws = torch.ones(256, 2, 16, device="cuda", dtype=torch.float32)
c = torch.empty(1, 8, 256, device="cuda", dtype=torch.bfloat16)
tw = torch.ones(1, 8, device="cuda", dtype=torch.float32) / 8
ti = torch.randint(0, 256, (1, 8), device="cuda", dtype=torch.int32)
srt, eid, npp = ... # from moe_align_block_size (正常大小)

# ✅ OK: mul_routed_weight=True (baseline)
_G(x, w, c, None, ws, None, tw, ti, srt, eid, npp, True, 8, 16,128,128,
   True, False, False, False, [128,128], 1)

# ❌ CRASH: mul_routed_weight=False (仅改变这一个参数)
_G(x, w, c, None, ws, None, tw, ti, srt, eid, npp, False, 8, 16,128,128,
   True, False, False, False, [128,128], 1)
# → static_assert(K % VEC_K == 0)
```

### 根因假设
DLEOL JIT 的 `mul_routed_weight` 模板特化导致 auto-tuner 选择了不兼容 K 维度的 VEC_K。
K=128（w2 的 intermediate dim）或 K=2048（w1 的 hidden dim），VEC_K 可能被选为不整除 K 的值。

### 建议修复
1. JIT auto-tuner 应在 VEC_K 不整除 K 时回退到 1（标量模式）
2. 或在 `invoke_fused_moe_opt` 的 C++ 层硬编码 VEC_K = gcd(VEC_K_candidates, K)

---

## Bug 2: Device page fault（JIT 运行时）

### 症状
```
[hal2]Failed to commit commands on device N: Device page fault
```

### 触发条件
1. `invoke_fused_moe_opt` 的 `use_moe_cu` 模式（trivial sorted_token_ids）
2. 新的 triton kernel 变体（如 silu_and_mul、fused_gdn_gating 输出 dtype 变化）

### 关键发现：独立测试通过，模型上下文崩溃

| 上下文 | use_moe_cu | 结果 |
|---|---|---|
| 独立脚本（无 torch.distributed） | ✅ OK（所有 topk 模式） | 3/3 通过 |
| 模型 TP2（有 torch.distributed） | ❌ Device page fault | 崩溃 |
| 模型 TP4（有 torch.distributed） | ❌ Device page fault | 崩溃 |
| 独立脚本 + 100 GEMM 预热 | ✅ OK | 通过 |

### 最小复现
```python
import torch
torch.ops.load_library("vllm/_dl_C.cpython-312-x86_64-linux-gnu.so")
_G = torch.ops._dl_C.invoke_fused_moe_opt

# 独立测试（OK）
x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
w = torch.randn(256, 256, 2048, device="cuda").to(torch.float8_e4m3fn)
ws = torch.ones(256, 2, 16, device="cuda", dtype=torch.float32)
c = torch.empty(1, 8, 256, device="cuda", dtype=torch.bfloat16)
tw = torch.ones(1, 8, device="cuda", dtype=torch.float32) / 8
ti = torch.randint(0, 256, (1, 8), device="cuda", dtype=torch.int32)
srt = torch.empty((1,), dtype=torch.int32, device="cuda")  # trivial
eid = torch.empty((1,), dtype=torch.int32, device="cuda")
npp = torch.empty((1,), dtype=torch.int32, device="cuda")

_G(x, w, c, None, ws, None, tw, ti, srt, eid, npp, False, 8, 16,128,128,
   True, False, False, False, [128,128], 1)
torch.cuda.synchronize()  # ✅ OK standalone

# 在 torch.distributed 初始化后的进程中运行同样代码 → ❌ Device page fault
```

### 根因假设
`torch.distributed` 初始化（NCCL process group）改变了 CUDA 上下文的某些属性
（可能是 stream 优先级、memory pool 布局、或 CUDA context flag），导致 DLEOL JIT
编译出的 kernel 在访问 use_moe_cu 的内部 dispatch 内存时越界。

### 建议修复
1. 在 DLEOL JIT 的 kernel 内部添加 bounds check（debug 模式）
2. 检查 use_moe_cu 的 dispatch 路径是否依赖 CUDA context 属性
3. 对比 torch.distributed init 前后的 CUDA context 差异

---

## Bug 3: `cudaErrorInvalidAddressSpace`（CG 捕获时）

### 症状
```
CUDA error: operation not supported on global/shared address space
```

### 触发条件
Bug 2 的 use_moe_cu + CUDA graph 捕获模式。

### 与 Bug 2 的关系
Bug 2 是 eager 模式下的 Device page fault；Bug 3 是 CG 模式下的
cudaErrorInvalidAddressSpace。两者可能是同一根因（CUDA context 状态差异）
在不同模式下的表现。

### 测试过的 CG 变体（全部崩溃）
- `capture_error_mode="global"`（默认）
- `capture_error_mode="relaxed"`
- `cuda_graph_backend_decode="tc_piecewise"`

---

## Bug 4: 加载 vLLM `_C.so` 破坏 DLEOL JIT 状态

### 症状
加载 `vllm/_C.cpython-312-x86_64-linux-gnu.so` 后，DLEOL JIT 对已成功编译的
baseline key 也触发 `K % VEC_K == 0` assert。

### 最小复现
```python
import torch

# Step 1: 先加载 _dl_C 并成功编译 baseline key
torch.ops.load_library("vllm/_dl_C.cpython-312-x86_64-linux-gnu.so")
_G = torch.ops._dl_C.invoke_fused_moe_opt
# ... 运行 baseline（OK）

# Step 2: 加载 vLLM _C（注册新的 custom ops）
torch.ops.load_library("vllm/_C.cpython-312-x86_64-linux-gnu.so")

# Step 3: 再次运行 baseline（同样的参数）
# → ❌ K % VEC_K == 0 assert
```

### 根因假设
vLLM `_C.so` 的加载过程注册了新的 triton/custom op backends，改变了 DLEOL JIT
的全局编译配置（如默认 VEC_K 或 auto-tuning 规则），导致后续 JIT 编译产生
不兼容的 kernel。

### 建议修复
1. DLEOL JIT 的 auto-tuning 配置应该是 per-op 的，不受其他 .so 加载影响
2. 或在 `_C.so` 加载时保存/恢复 DLEOL JIT 配置

---

## 影响

这些 bug 阻止了以下优化（每项都已验证在独立测试中可用）：

| 优化 | 预期节省 | 阻塞 bug |
|---|---|---|
| use_moe_cu（跳过 MoE align） | ~2ms/step | Bug 2+3 |
| fused silu_and_mul（1 kernel vs 2） | ~0.5ms/step | Bug 1+4 |
| bf16 beta output（省 30 个 .to kernel） | ~0.3ms/step | Bug 2 |
| mul_routed_weight=False（match vLLM w2） | ~0.5ms/step | Bug 1 |
| **合计** | **~3.3ms/step** | |

加上 JIT 变体对齐（~5ms，需架构调整），可关闭大部分 9.2ms gap。

---

## 验证环境

```bash
source ../sdk/env.sh  # dl19-matching libs
source .venv/bin/activate  # sglang venv
# 模型路径
MODEL=/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8
# vLLM venv（_dl_C.so 和 _C.so 来源）
VLLM_VENV=../venv-vllm021
```

---

## 联系

- `docs/dl/sglang-vllm-tp4-gap-report.md`（10 章完整报告）
- `docs/dl/sglang-vs-vllm-perf-gap.md` §7.24–7.29（逐轮实验记录）
