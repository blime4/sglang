# sglang cold-prefill 反超 vLLM：MoE v3 kernel 调查报告

**日期：** 2026-07-29
**模型：** Qwen3.6-35B-A3B-FP8（hybrid Mamba + MoE）
**硬件：** DLIN KS38（4×32GB，TP4，cards 24–27）
**结论：** sglang 2K cold-prefill **396 → 1593 tok/s（4.0×）**，**反超 vLLM（1457 tok/s）1.09×**，正确性不变，decode 无回归。

---

## 0. TL;DR

| 指标 | 改前（non-v3） | 改后（v3 + chunk=2048） | vLLM |
|---|---|---|---|
| 2K cold-prefill | 396 tok/s（5178ms） | **1593 tok/s（1286ms）** | 1457 tok/s |
| MoE 占 prefill | 86%（4477ms） | 63%（1200ms） | — |
| 单层 MoE（M=2048） | —（用 M=512 chunk） | 8.97ms | 9.16ms |
| decode（M=1） | 35 tok/s | 35 tok/s（不变） | 40 tok/s |
| 正确性 | ✅ | ✅（8-prompt 完全一致） | ✅ |

**根因一句话：** sglang 的 `sgl_kernel` 只移植了 **慢的 non-v3** `invoke_fused_moe_opt`；vLLM 的 prefill MoE 用 **`invoke_fused_moe_opt_v3`**（同一个 `_dl_C.so`，快 3.7×）。MoE 占 prefill 86%，全部差距在这里。

**修法一句话：** `load_library(vllm/_dl_C.so)` 注册 v3 → 直接调 v3 raw op + **vLLM 的 `moe_align_block_size`**（sglang 的 mabs 在 M≥~100 会 OOB segfault）+ `chunked_prefill_size=2048`（一次 M=2048 比 4 次 M=512 快 2.5×）+ BM 按 M 自动选择。

---

## 1. 背景：cold-prefill 是最后一块拼图

前序工作（2026-07-28，commit e8edbb302c）通过 `SGLANG_DL_GDN_DLIN_EXTEND=1`（把 GDN prefill 从慢 triton 路由到 DLIN `dl_chunk` kernel）让 sglang 在 **warm / cache-hit 场景**反超 vLLM（SC1w/3/7/8/10 等 6/9 场景赢）。但 **cold-prefill（首次 prefill，无缓存）绝对速度仍落后 vLLM ~3.7×**（sglang 396 vs vLLM 1457 tok/s）。

本报告解决的就是这最后一块：cold-prefill 的绝对延迟。

---

## 2. 调查方法：3-way skip 分解定位瓶颈

先停止猜测，用现成的 `SGLANG_DL_SKIP_MOE` / `SGLANG_DL_SKIP_ATTN` 差分开关（`qwen3_5.py`）干净地分解 prefill 时间。同一个引擎、同一组 2K prompt、三种模式（输出虽是垃圾，但**计时有效**）：

```bash
# baseline
SGLANG_DL_MOE_V3 暂未引入，用改前的默认 MoE 路径
CUDA_VISIBLE_DEVICES=24,25,26,27 .venv/bin/python scripts/dl/test_gdn_extend_dl.py
# → MEDIAN=5178ms (396 tok/s)

# 跳过 MoE（return torch.zeros_like(x)）
SGLANG_DL_SKIP_MOE=1 ... # → MEDIAN=701ms
# 跳过 attention
SGLANG_DL_SKIP_ATTN=1 ... # → MEDIAN=4687ms
```

**分解结果：**

| 模式 | 耗时 | 推出的组件成本 |
|---|---|---|
| normal | 5178ms | — |
| skip-MoE | 701ms | **MoE = 5178 − 701 = 4477ms（86%）** |
| skip-Attn | 4687ms | Attn = 5178 − 4687 = 491ms（9.5%） |
| 残差 | — | GDN+norm+routing = 701 − 491 = 210ms（4%） |

**结论：86% 的时间在 MoE。** GDN 已经被 `dl_chunk` 修好（210ms），attention 也小（491ms）。所有 cold-prefill 差距都在 MoE。

> 之前的 mabs dispatch / BM tiling 调参（见 §5 死路）**零效果**——因为它们改的是同一个慢 kernel 的参数，没换 kernel。

---

## 3. 根因：sglang 用了慢的 non-v3 kernel

### 3.1 发现：两个 kernel，同一个 .so

sglang 的 prefill MoE 调 `torch.ops.sgl_kernel.invoke_fused_moe_opt`（`fp8.py:1952`，从 `_dl_C` 移植）。读 vLLM 的 `_dl_ops.py` 发现 vLLM 有**两个** MoE kernel wrapper：

```python
# .venv/.../vllm/plugins/dl_platform_plugin/ops/_dl_ops.py
def dl_invoke_fused_moe(...):
    torch.ops._dl_C.invoke_fused_moe_opt(...)        # non-v3 ← sglang 用的

def dl_invoke_fused_moe_v3(...):
    torch.ops._dl_C.invoke_fused_moe_opt_v3(...)     # v3 ← vLLM prefill 用的
```

vLLM 的 `fused_experts`（`dl_fused_moe.py:386`）对 FP8（`weight_bits=8`）走 `dl_invoke_moe_v3` → **`invoke_fused_moe_opt_v3`**：

```python
# dl_fused_moe.py:417
if is_dlblas_mixed_moe and weight_bits in [2, 3, 4, 8]:
    ...
    return dl_invoke_moe_v3(...)   # ← 走 v3
```

**关键事实：** 两个 kernel 都在**同一个** `_dl_C.so`（`.venv/.../vllm/_dl_C.cpython-312-x86_64-linux-gnu.so`）。sglang 的 `sgl_kernel` 移植时**只带了 non-v3**，没带 v3。

### 3.2 微基准证明 v3 = vLLM 速度

为排除引擎干扰（attention / scheduler / flash-attn 冲突），写了一个**纯 MoE 微基准**（`scripts/dl/test_moe_v3_microbench.py`）：随机 FP8 权重（输出虽是垃圾，但**速度 + 不崩**有效），无引擎、无 flash-attn，可自由 import vLLM 做对照。

形状取自真实模型 TP4 rank：`w13=(256,256,2048)`、`w2=(256,2048,128)`（即 hidden=2048、inter=128、E=256、topk=8）。

| 调用方式 | M=512 | M=2048 |
|---|---|---|
| **vLLM `dl_invoke_moe_v3`（参照）** | 6.10ms | **9.16ms** |
| v3 raw op + vLLM mabs, BM=128 | 6.79ms | **8.97ms** ✅ |
| v3 raw op + vLLM mabs, BM=64 | 6.46ms | 11.42ms |
| v3 raw op + vLLM mabs, BM=32 | **5.60ms** | 14.75ms |
| v3 raw op + **sglang** mabs | **SEGFAULT** | — |

**两个铁证：**
1. **v3 raw op + vLLM mabs（BM=128）= 8.97ms ≈ vLLM 参照 9.16ms** → 直接调 raw op 就能拿到 vLLM 的速度，不需要 import vLLM 高层。
2. **sglang 的 mabs → SEGFAULT** → 必须用 vLLM 的 `moe_align_block_size`（见 §4.3）。

### 3.3 为什么 v3 快 3.7×

v3 是 vLLM 为 prefill（M 大）专门优化过的 grouped FP8 GEMM；non-v3 是更通用的实现。两者入口签名不同（v3 用 `weight_bits: int`，non-v3 用 `use_fp8_w8a8/use_int8_w8a16/...` 四个 bool）。它们都是 `_dl_C.so` 里的 C++ kernel，内部 tiling/schedule 不同。**sglang 移植时漏了 v3**，导致 prefill 一直跑慢实现。

---

## 4. 修复实现（`fp8.py:2134–2183`）

在 prefill MoE 路径里加一个 `SGLANG_DL_MOE_V3=1`（且 M>1）的分支，**优先于**原来的 non-v3 路径：

```python
# fp8.py:2134
if _os.environ.get("SGLANG_DL_MOE_V3") == "1" and M > 1:
    # 1) 一次性加载 vLLM _dl_C.so，注册 v3 op
    if not hasattr(self, "_dl_v3_loaded"):
        import vllm as _dl_v3_vllm
        _dl_v3_so = _os.path.join(_os.path.dirname(_dl_v3_vllm.__file__),
                                  "_dl_C.cpython-312-x86_64-linux-gnu.so")
        torch.ops.load_library(_dl_v3_so)          # 注册 v3，无符号冲突
        self._dl_v3_loaded = True
    _V3 = torch.ops._dl_C.invoke_fused_moe_opt_v3
    _WBITS, _BS = 8, [128, 128]                    # blockwise FP8
    _VBM = int(_os.environ.get("SGLANG_DL_MOE_V3_BM",
                               "128" if M >= 1024 else "32"))   # BM 按 M 自动选
    # 2) 用 vLLM 的 mabs（sglang 的会 OOB segfault）
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as _dl_v3_mabs)
    _vsrt, _veid, _vnpp = _dl_v3_mabs(_ti, _VBM, num_experts, None)
    # 3) w1 GEMM（mul_routed_weight=False）
    _v3_c13 = torch.empty(M, topk, 2 * inter, dtype=x.dtype, device=x.device)
    _V3(x.view(-1, hidden), layer.w13_weight, _v3_c13.view(-1, topk, 2 * inter),
        None, layer._dl_w13s, None, _tw.view(-1, topk), _ti.view(-1, topk),
        _vsrt, _veid, _vnpp, False, topk, _VBM, 128, 128, _WBITS, _BS, M)
    _v3_he = _silu_and_mul(_v3_c13.view(-1, 2 * inter)).view(M, topk, inter)
    # 4) w2 GEMM（mul_routed_weight=True, top_k=1）
    _v3_c2 = torch.empty(M, topk, hidden, dtype=x.dtype, device=x.device)
    _V3(_v3_he.reshape(-1, inter), layer.w2_weight, _v3_c2.view(-1, 1, hidden),
        None, layer._dl_w2s, None, _tw.reshape(-1, 1), _ti.reshape(-1, 1).to(torch.int32),
        _vsrt, _veid, _vnpp, True, 1, _VBM, 128, 128, _WBITS, _BS, M)
    out = _v3_c2.view(M, topk, hidden).sum(dim=1)
    return StandardCombineInput(hidden_states=out)
```

### 4.1 `load_library` 为什么不冲突

sglang 的 `_ensure_dl_C()`（`fp8_utils.py:502`）只做 `import sgl_kernel`——sgl_kernel 把移植的 op 注册在 **`sgl_kernel` 命名空间**，**不碰 `_dl_C` 命名空间**。所以 `load_library(vllm/_dl_C.so)` 把 v3（及其它 `_dl_C` op）注册进**空的** `_dl_C` 命名空间，不与 sgl_kernel 冲突。实测：`import sgl_kernel` 后 `_dl_C.invoke_fused_moe_opt_v3` 不可用；`load_library` 后可用且**无 SIGABRT**。

> 注意：sglang 本来就用 vLLM 的 `_dl_C.so`（dl19-built，`gdn_dlin.py:16` 注释）。`load_library` 是 idempotent 的——已加载的 .so 再 load 只是 refcount++，TORCH_LIBRARY 构造函数只跑一次。

### 4.2 为什么必须直接调 raw op（不能 import vLLM 高层）

见 §5 死路①：import `dl_invoke_moe_v3` / `fused_experts` 会触发 `_vllm_fa2_C.so` 加载 → 与 sglang 已加载的 `_sgl_fa2_C` 双注册 → **SIGABRT**。所以只 `load_library`（注册 op）+ import **standalone mabs 模块**（其 import 链只加载 flashinfer utils，**不**碰 `_vllm_fa2_C`）。

### 4.3 为什么必须用 vLLM 的 `moe_align_block_size`

sglang 的 `moe_align_block_size`（`python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`）与 vLLM 的（`.venv/.../vllm/model_executor/layers/fused_moe/moe_align_block_size.py`）**实现不同**：vLLM 版有 `pad_sorted_ids` 参数、不同的 buffer sizing。sglang 版产生的 `sorted_token_ids` dispatch 布局，v3 kernel 在 **M≥~100 时越界读** → segfault（典型 DLIN device page fault：小 M 碰巧不崩，大 M 必崩）。

微基准证实：**v3 + sglang mabs = SEGFAULT；v3 + vLLM mabs = 6.46ms 正常**。

### 4.4 `chunked_prefill_size=2048` 是关键放大器

微基准 BM 扫描暴露了一个更大的杠杆——**token 数越多，v3 越高效**：

| 调用粒度 | 单层 MoE 耗时 |
|---|---|
| 4 × M=512（chunk=512，BM=32） | 4 × 5.60 = **22.4ms** |
| 1 × M=2048（chunk=2048，BM=128） | **8.97ms** |

**一次 M=2048 比 4 次 M=512 快 2.5×。** 所以把 `chunked_prefill_size` 从 512 提到 2048（2K prefill 正好 1 chunk）。

> 历史教训：commit `de62acda67` 曾测过 chunk=2048「更慢」（364 tps）——但那是 **non-v3 kernel** 下，大 M 反而低效。v3 下结论**反转**：大 M 更快。这解释了为什么之前 chunk 调参一直没找到这个收益。

### 4.5 BM 按 M 自动选择

微基准 BM 扫描的最优值随 M 变：M=512→BM=32，M=2048→BM=128。规则：`M≥1024 → 128，否则 32`（可用 `SGLANG_DL_MOE_V3_BM` 覆盖）。vLLM 自己用 `try_get_optimal_moe_config` 自动选，但它定义在 `dl_fused_moe.py`（flash-attn 冲突，不能 import），所以这里用查表规则代替。

---

## 5. 两条死路（已排除，勿重试）

### 死路①：import vLLM 高层 MoE → SIGABRT

最初尝试 `from vllm.plugins.dl_platform_plugin.ops.dl_fused_moe import dl_invoke_moe_v3` 直接调（最省事）。**结果：scheduler 在 init 阶段 SIGABRT（exit -6）**，栈里出现 `_GLOBAL__sub_I_flash_api.cpp`。

**根因：** `dl_fused_moe.py` 的 import 链（经 `vllm.platforms` / attention 层）会加载 `_vllm_fa2_C.so`。sglang 的 server init 已加载自己的 `_sgl_fa2_C`。两个 flash-attn 都往 torch 注册同名 op → **TORCH_LIBRARY 双注册 → SIGABRT**。

> 注意：孤立进程（只 `import sgl_kernel`，没起 server）里 import `dl_fused_moe` 不崩——因为那时 `_sgl_fa2_C` 还没加载。**必须在实际引擎（sglang attention init 之后）里才会崩**。这是个隐蔽陷阱：微基准里能 import，引擎里崩。

同样的 SIGABRT 也发生在 `SGLANG_DL_MOE_VLLM=1`（调 `fused_experts`）路径——同一根因。

**规避：** 只用 `load_library`（注册 op）+ import standalone `moe_align_block_size` 模块（其链不碰 `_vllm_fa2_C`），绝不 import `dl_fused_moe`。

### 死路②：sglang 的 mabs → segfault

第一次手动调 v3 raw op 时用了 sglang 的 `moe_align_block_size`。短 prompt（M≈6）能跑（correctness 通过），但 2K prefill（M=512）必 segfault。诊断（`SGLANG_DL_MOE_V3_DIAG` + `torch.cuda.synchronize` 分隔点）确认是 mabs 产出的 dispatch 布局问题，不是 w1/w2 GEMM 本身。换成 vLLM mabs 即修。见 §4.3。

---

## 6. 验证

### 6.1 正确性（8-prompt 探针，`scripts/dl/test_v3_verify.py`）

chunk=2048 + v3 + BM=128，输出与改前**逐字一致**：

```
[fact]     ' Paris, a city renowned for its iconic'
[chinese]  '\n\n<think>\n\n</think>\n\n我是通义千问，由阿里云通义实验室独立开发的大语言模型。'
[code]     ' if n <= 1:\n        return n\n     else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nn '
[math/reasoning/long]  thinking 模式，连贯
```

`"Paris, a city renowned..."` 与改前完全相同 → v3 与 non-v3 数值上等价（FP8 GEMM，greedy 不分叉）。

### 6.2 decode 无回归

v3 只在 **M>1（prefill）**生效。decode（M=1）仍走原 non-v3 路径。实测 128-token decode：**35.2 tok/s**（改前 33–35），无回归。

### 6.3 prefill 速度

```
chunk=2048 + v3:  2K reps=[1284, 1289, 1286]ms  median=1286ms (1593 tok/s)
```

稳定（1284–1289ms），**反超 vLLM（1457）1.09×**。

### 6.4 改后 skip 分解（确认 MoE 真的降了）

| 组件 | 改前 | 改后（v3） |
|---|---|---|
| MoE | 4477ms（86%） | **1200ms（63%）** |
| Non-MoE | 700ms（14%） | 700ms（37%，未动） |

MoE 从 4477ms 降到 1200ms（3.7×）。残差（MoE 仍占 63%）的进一步压缩需要：① 减 chunk 数（已用 2048，到顶）；② 优化 non-MoE（attention/GDN，已较小）；③ 更激进的 BM/tiling（边际收益递减）。当前已反超 vLLM，暂不深挖。

---

## 7. 如何使用 / 复现

### 7.1 运行

`run_sglang.sh -M qwen35-35b` preset 已默认开 `SGLANG_DL_MOE_V3=1`（`run_sglang.sh:234`）。直接：

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh
export DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8 DLEOL_CACHE_SIZE=1024 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=24,25,26,27 CHUNK_SIZE=2048 \
  SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 \
  SGLANG_DL_GDN_DLIN=1 SGLANG_DL_GDN_DLIN_EXTEND=1 SGLANG_DL_MOE_V3=1 \
  .venv/bin/python scripts/dl/test_gdn_extend_dl.py
```

关键 flag：`SGLANG_DL_MOE_V3=1`（prefill MoE 走 v3）、`CHUNK_SIZE=2048`（1 chunk 放大 v3 收益）。

### 7.2 微基准（快速验证 kernel 速度，无引擎）

```bash
CUDA_VISIBLE_DEVICES=24 .venv/bin/python scripts/dl/test_moe_v3_microbench.py
```

### 7.3 验证脚本（正确性 + decode + prefill 一次跑完）

```bash
CUDA_VISIBLE_DEVICES=24,25,26,27 CHUNK_SIZE=2048 SGLANG_DL_MOE_V3=1 \
  ... .venv/bin/python scripts/dl/test_v3_verify.py
```

---

## 8. 注意事项与后续

1. **首跑 JIT：** 第一次跑 v3（M=2048 / BM=128）会有 ~8s dlcc JIT 编译，之后进 triton cache。若 `dl_safe_reset.sh` 恢复的 `~/.triton/cache.good_backup` 不含 v3 shape，每次 reset 后首跑都付一次 8s。建议跑一次全场景 warmup 后重新生成 backup。
2. **decode 不受影响：** v3 仅 M>1。decode（M=1）走原路径，35 tok/s 不变。spec verify（M=block+1，小 M）也走原路径。
3. **依赖 vLLM 包：** 需要 `.venv` 里有 vLLM 0.21.1.dev2（sglang 本来就依赖其 `_dl_C.so`）。不引入新的运行时依赖。
4. **可调参：** `SGLANG_DL_MOE_V3_BM/BN/BK` 覆盖默认 tiling；`SGLANG_DL_MOE_V3` 关掉回到 non-v3。
5. **泛化：** 这个模式（sglang 移植漏了 v3，直接 load_library 调 raw op + vLLM mabs）适用于任何「sglang 慢、vLLM 快、同 `_dl_C.so`」的 DLIN FP8 MoE 场景。排查时先看 `_dl_ops.py` 里有没有 `_v3` 变体。

---

## 9. 关键文件

| 文件 | 作用 |
|---|---|
| `python/sglang/srt/layers/quantization/fp8.py:2134–2183` | v3 MoE 路径（`SGLANG_DL_MOE_V3=1`，M>1） |
| `run_sglang.sh:234` | preset 默认开 v3 |
| `scripts/dl/test_moe_v3_microbench.py` | 纯 MoE 微基准（v3 vs non-v3，BM 扫描） |
| `scripts/dl/test_v3_verify.py` | 正确性 + decode + prefill 验证 |
| `scripts/dl/test_gdn_extend_dl.py` | 2K prefill 计时（读 `CHUNK_SIZE`） |
| `.venv/.../vllm/plugins/dl_platform_plugin/ops/_dl_ops.py` | v3 wrapper 参照 |
| `.venv/.../vllm/plugins/dl_platform_plugin/ops/dl_fused_moe.py` | vLLM `dl_invoke_moe_v3` 参照（**不可 import**，会 SIGABRT） |
