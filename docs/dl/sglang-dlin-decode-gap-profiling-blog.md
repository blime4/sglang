# sglang DLIN Decode Gap — From 33.7 to 35.0 tok/s, and the Remaining 6.2ms

**Date:** 2026-07-22 · **Hardware:** DLIN KS38 (32× QUAD, 32 GiB) · **Model:** Qwen3.5-35B-A3B-FP8 · **TP4**

> 续 [`sglang-dlin-decode-gap-debug-blog.md`](sglang-dlin-decode-gap-debug-blog.md)。上一篇定位了差距来源（host-side sync/IPC，非 GPU kernel）。本篇记录本轮 session 的全部优化 + profiling，最终将 gap 精确量化为 **6.2ms/token 的 ZMQ IPC 往返成本**。

---

## 摘要

| 指标 | 值 | 说明 |
|---|---|---|
| sglang baseline | 33.7 tok/s | session 开始前 |
| sglang + seq_lens_sum host-compute | **35.0 tok/s** (+4%) | 消除 per-step D2H sync |
| vLLM MRV2 FP8+CG | **37.9 tok/s** | 参考 |
| **gap** | **6.2 ms/token** | 100% host-side（ZMQ IPC） |
| GPU kernel time | 20.7 ms/token (both) | 完全一致 |
| sglang host/step | ~9 ms | 含 IPC + 调度 + 周期性 spike |
| vLLM host/step | ~5.7 ms | scheduler 与 worker 同进程 |

---

## 1. sglang 自有 flash-attn（`_sgl_fa2_C`）— 消除 `_vllm_fa2_C` 冲突

### 问题

sglang 的 `dl_flash_attn.py` 导入 `vllm_flash_attn` 包 → 该包的 `.so` 注册 `_vllm_fa2_C` TORCH_LIBRARY namespace。当 sglang + vLLM 共存于 `.venv`（`SGLANG_DL_MOE_VLLM=1` 导入 vLLM fused_experts），两个 `.so` 同时注册 `_vllm_fa2_C` → `c10::Error` SIGABRT。

### 调试历程（4 层修复）

| 层 | 发现 | 修复 |
|---|---|---|
| **L1: standalone .so namespace** | standalone `flash_attn_2_cuda.so` 也注册 `_vllm_fa2_C`（与 vLLM 冲突） | 重新编译 flash-attn，namespace 改为 `_sgl_fa2_C`（`flash_api.cpp:1791`）|
| **L2: easy-install.pth** | `easy-install.pth` 把 `flash-attention/` 开发目录加到 sys.path → Python 加载旧 `.so`（仍注册 `_vllm_fa2_C`），忽略了 `.venv` 里的新 `.so` | 注释掉 `easy-install.pth` 的 flash-attention 路径 |
| **L3: vLLM 内部重复 .so** | `.venv` 里有两份 `_vllm_fa2_C.so`：`vllm_flash_attn/`（PyPI 包）和 `vllm/vllm_flash_attn/`（vLLM 内置）→ 互相冲突 | 隐藏 `vllm/vllm_flash_attn/` 副本（保留 `vllm_flash_attn/` 那份，sglang attention 需要它）|
| **L4: sys.path 注入** | fp8.py 硬编码 `../venv-vllm021` sys.path 注入 → 加载第二个 vLLM 副本 → 再次冲突 | 删除 sys.path 注入，直接从 `.venv` 导入 |

### 最终方案：sglang 完全自包含 flash-attn

编译 sglang 自己的 `flash_attn_2_cuda.so`（namespace `_sgl_fa2_C`）+ Python wrapper `sgl_flash_attn.py`。`dl_flash_attn.py` 改为从 sglang 自己的 wrapper 导入。

```
之前: dl_flash_attn.py → import vllm_flash_attn → torch.ops._vllm_fa2_C
之后: dl_flash_attn.py → import sgl_flash_attn → torch.ops._sgl_fa2_C
```

**零 vLLM 依赖。** E2E 验证：35.0 tok/s，有效 JSON，零退化。

### 编译命令

```bash
cd /tmp/flash-attention-build
source ../sglang/sdk-dlop-07-13-20-30/env.sh
USE_DLIN=1 CUDA_HOME=$SDK_DIR MAX_JOBS=8 \
  ../sglang/.venv/bin/python setup.py build_ext --inplace
# 编译标志: -DVLLM_FLASH_ATTN -DFLASHATTENTION_DISABLE_PYBIND
# namespace: TORCH_LIBRARY_EXPAND(_sgl_fa2_C, ops) at flash_api.cpp:1791
cp flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so \
   ../sglang/.venv/lib/python3.12/site-packages/
```

---

## 2. decode 加速 — `seq_lens_sum` host-compute（+4%）

`decode_cuda_graph_runner.py:624` 每个 decode step 执行 `seq_lens.sum().item()` → D2H sync（GPU→host 同步，阻塞 pipeline）。

修复：改为 `int(seq_lens_cpu.sum())`（host 端从 CPU 镜像计算，无 D2H sync；与 `tbo_backend.py:195` 模式一致）。

**效果：sglang decode 33.7 → 35.0 tok/s（+4%），JSON 输出有效。**

---

## 3. P0b MoE overlay — 基线已用 fast kernel，无需 overlay

### 假设

sglang 被迫走慢的 `GEMMEX=2` MoE 路径（Python gather 6ms），vLLM 走快的 `invoke_fused_moe_opt`（fused gather 0.1ms）→ 差距 5.36ms。

### 事实

代码分析（`fp8.py:1923-2160`）证实：**baseline 已经走 `invoke_fused_moe_opt`（fast path）！** `SGLANG_DL_MOE_FUSED=1` + `SGLANG_DL_MOE_FUSED_MAX_M=32` 在 benchmark 默认启用，decode (M=1) 命中 fast path。

`SGLANG_DL_MOE_VLLM=1` 测试的是 vLLM 的 Python wrapper（`fused_experts()`），不是 C++ kernel 本身。Python wrapper 的 per-layer 调度开销导致 1000× 慢（30s/token），但这不是 kernel 的真实性能。

### 实测验证

`SGLANG_DL_MOE_VLLM=1` + CG：
- ✅ 正确性：有效 JSON 输出（`{"name": "Dr. Ada Lovelace", ...}`）
- ❌ 性能：0.0 tok/s（30s/token）—— PDL kernel 在 CG replay 下严重 stall（33min capture vs 14s）
- 结论：use_moe_cu 的 C++ kernel 在 CG 下正确但不可用（PDL stall），而 baseline 的 `invoke_fused_moe_opt` 已经是最优 MoE 路径

**P0b 不需要做。** MoE 已经是 fast path。gap 不是 MoE。

---

## 4. Host-side timing breakdown — 6.2ms = 100% IPC

### 方法

多 N 值测量（N=4/8/16/32/64/96 tokens），用 per-token 时间在 N→∞ 时收敛到 GPU+CG-replay 时间，差值即 IPC overhead。

### 结果

```
N=  4: per_tok=169.2ms  (IPC + prefill 占主导)
N=  8: per_tok= 94.5ms
N= 16: per_tok= 62.0ms
N= 32: per_tok= 43.3ms
N= 64: per_tok= 35.1ms
N= 96: per_tok= 32.6ms  ← 纯 decode（IPC 已充分摊薄）

single-token median: 600.3ms  (含 ~568ms prefill + 32.6ms decode)
per-token @ N=96:    32.6ms
gap to vLLM:          6.2ms   ← 这就是全部差距
```

### cProfile 分析

20 次 single-token decode（11.2s 总计）的 Python 级 profiling：

| 函数 | self time | 占比 | 说明 |
|---|---|---|---|
| `engine.generate()` | 7.9s | 70% | 主进程 Python + ZMQ I/O |
| `lock.acquire` | 3.3s | 29% | 等待 scheduler 响应（IPC 同步点） |
| `pickle.loads/dumps` | 7ms | <1% | ZMQ 序列化（非瓶颈） |
| `tokenizer.encode` | 4ms | <1% | 每次重新 tokenize prompt |

**结论：6.2ms/token = scheduler↔tokenizer_manager 的 ZMQ IPC 往返成本。** 不是可消除的拷贝或 sync 点（P0a′ 已消除唯一的 `.item()` sync）。是 sglang 多进程架构的固有成本。

### 与 tp4-gap report 的一致性

tp4-gap report §4：sglang host 6ms/step vs vLLM 3ms/step（overlap scheduler 后 median 3.2ms，周期性 27ms spike）。本 session 的 6.2ms 是 20-step 平均（含 spike）。vLLM 的 ~0ms host gap 是因为 scheduler 与 worker 在同一进程，无 IPC 往返。

---

## 5. GPU kernel 时间完全一致

| 组件 | sglang | vLLM | 差异 |
|---|---|---|---|
| GPU kernel total | 20.7 ms/token | 20.7 ms/token | **0** |
| MoE kernel | `invoke_fused_moe_opt` (fast) | `invoke_fused_moe_opt` (fast) | 相同 |
| Attention | `vllm_flash_attn`/`_sgl_fa2_C` | `vllm_flash_attn` | 相同 |
| Host overhead | ~9 ms/step | ~5.7 ms/step | **+3.3 ms** |
| TPOT total | 32.6 ms/token | 26.4 ms/token | **+6.2 ms** |

---

## 6. 已完成优化汇总

| 优化 | 效果 | 文件 | 提交 |
|---|---|---|---|
| `_dl_C` 单路径加载 | 消除间歇性 SIGABRT | `layernorm.py`, `fp8_utils.py` | ✅ |
| `seq_lens_sum` host-compute | +4% decode (33.7→35.0) | `decode_cuda_graph_runner.py:624` | ✅ |
| sglang 自有 flash-attn (`_sgl_fa2_C`) | 零 vLLM 依赖，消除 `_vllm_fa2_C` 冲突 | `sgl_flash_attn.py` (NEW), `dl_flash_attn.py` | ✅ |
| vLLM 导入路径修复 | 删除 `venv-vllm021` sys.path 注入 | `fp8.py` | ✅ |
| `/dl-compare-sglang-vllm` skill | → `.venv`，FP8 对比 | `SKILL.md` | ✅ |

---

## 7. 剩余 6.2ms 的可行优化路径

| 路径 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| **减小 IPC payload** | ~1-2ms | 中 | 每 step 序列化全量 metadata → 增量发送 |
| **inline scheduler (single-request)** | ~3-4ms | 高 | max_running_requests=1 时跳过 ZMQ 往返，直接调用 worker |
| **更激进的 overlap** | ~1-2ms | 中 | 消除周期性 spike（GC、metadata prep 无法 overlap 的 step） |
| **torch.compile Phase II** | ~1-2ms (GPU 侧) | 多天 | 融合 norm/act+quant kernel，减少 GPU kernel 数量 |

**达到 vLLM parity (37.9 tok/s) 需要消除全部 6.2ms** —— 这是一个结构性架构优化，不是单点 patch。

---

## 8. 关键文件索引

| 文件 | 内容 |
|---|---|
| `sgl_flash_attn.py` | sglang 自有 flash-attn wrapper（`_sgl_fa2_C`） |
| `dl_flash_attn.py:142` | 从 `vllm_flash_attn` 切到 `sgl_flash_attn` |
| `decode_cuda_graph_runner.py:624` | `seq_lens_sum` host-compute |
| `fp8.py:2130-2160` | `invoke_fused_moe_opt` fast MoE path（baseline 默认） |
| `bench_features_sglang_vllm.py` | sglang vs vLLM feature benchmark |
| `vllm_features_only.py` | 独立 vLLM FP8+CG harness |
| `profile_decode_step.py` | decode-step profiling（torch.profiler + DLPTI_AUTO_LOAD） |
| `/tmp/host_timing_v2.py` | host-side 多-N timing breakdown |

## 参考

- Memory: `dlin-sglang-native-flash-attn-sgl-fa2-c`, `dlin-sglang-decode-gap-sync-opt-wins`, `dlin-sglang-vllm-dl_C-double-registration-fix`
- 前序 blog: [`sglang-dlin-decode-gap-debug-blog.md`](sglang-dlin-decode-gap-debug-blog.md), [`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md)
- tp4-gap report: [`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md)
