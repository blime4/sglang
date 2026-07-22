# 缩小 sglang 在 DLIN 上相对 vLLM 的解码差距 —— 调试实战记录

**日期：** 2026-07-21 · **硬件：** DLIN KS38（32× QUAD，32 GiB）· **模型：** Qwen3.5-35B-A3B-**FP8**（两个引擎都用）· **TP4，GPU 28–31**

> 目标：在 DLIN 上让 sglang-CG 解码 ≥ vLLM-CG（实测 sglang 33.7 tok/s vs vLLM MRV2 37.9）。本文记录这次尝试的调试过程——修了什么、排除了什么、以及精确的下一步计划。配套文章：[`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md)（JSON / 前缀共享对比）。

---

## 摘要（TL;DR）

- **修复了一个真实阻塞：** sglang 与 vLLM 共存在 `.venv` 后，sglang 初始化会间歇性 `SIGABRT`（`_dl_C` 双重注册）。修复后 sglang 可确定性初始化。
- **推翻了原先的核心假设：** 之前"≈21 次 eager 拷贝/步 = 解码差距"的说法是**错的**——完整热路径追踪表明，拷贝*数量*解释不了 ~7 ms/token 的差距。真正原因是 **GPU 同步点 + 多进程 IPC 的结构性开销**，而不是张量拷贝。
- **multi-step 方案已弃用（实测）：** `SGLANG_DL_MULTI_STEP` **慢约 4×，不是更快**（8.6 vs 33.7 tok/s——它的 `.item()` D2H 同步 + 投机分配/簿记开销占主导；之前"~2 ms/token"的说法在 DLIN 上不成立）。输出是对的，但这条路径得不偿失。其 KV 泄漏（`output % _dl_n == 0 → allocated` 翻倍）随之失去意义。
- **交付了首个实测加速：** 消除了一个每步 D2H 同步——`decode_cuda_graph_runner.py:624 seq_lens.sum().item()` → `int(seq_lens_cpu.sum())`（host 端计算，无同步；与 `tbo_backend.py:195` 一致）。**sglang 解码 33.7 → 35.0 tok/s（+4%），输出有效。**
- **实测：** 匹配 FP8+CG——sglang **35.0** vs vLLM MRV2 37.9 tok/s（从 33.7 → 差距由 ~12% 缩小到 ~8%）。MRV1 在 DLIN 上跑不起来。剩余差距是结构性的（scheduler↔worker IPC；需要 dlPTI profiling 找下一个同步点）。

---

## 1. 环境与匹配的基线

两个引擎都用 FP8、cuda-graph、TP4，同一台 DLIN 机器，temp 0，best-of-N。之前工作里的关键修正：vLLM 要用 **`.venv`**（vLLM 0.21.1.dev2 + DLIN triton 3.3.0，原生安装——**不用 overlay**）；旧的 `../venv-vllm021`+overlay 配置是坏的（见 [`dlin-vllm-correct-package-and-env-gotchas`](../../../home/shaobo.xie/.claude/projects/-LocalRun-shaobo-xie-2-Pytorch-docker-test-debug-sglang/memory/dlin-vllm-correct-package-and-env-gotchas.md)）。

| 引擎 | JSON tok/s（无约束→约束） |
|---|---|
| sglang FP8+CG | 33.7 → 31.7 |
| vLLM MRV2 FP8+CG | **37.9 → 34.6** |
| vLLM MRV1 FP8+CG | **失败**（`assert num_cache_lines >= batch`） |

两个引擎的 GPU kernel **字节完全一致**（都是 20.7 ms/token——同一个 `_dl_C.so`，md5 `adc7f6a2…`）。所以差距 **100% 在 host 侧**。

---

## 2. 已修复 —— `_dl_C` 双重注册 SIGABRT ⭐

### 现象
sglang 与 vLLM 共存于 `.venv` 后，sglang 在 cuda-graph 捕获时间歇性 `SIGABRT`（exit -6）：
```
c10::Error: Only a single TORCH_LIBRARY can be used to register the namespace _dl_C;
Previous registration at /vllm_workspace/vllm/csrc/dl/torch_bindings.cpp:14;
latest registration at .../vllm/csrc/dl/torch_bindings.cpp:14
```

### 根因
sglang 从**两个不同的文件系统路径**加载 `_dl_C.so` 自定义算子库：
- `python/sglang/srt/layers/layernorm.py:_dl_load_dl_C()` 和 `python/sglang/srt/layers/quantization/fp8_utils.py:_ensure_dl_C()` **硬编码**了 `../venv-vllm021/lib/python3.12/site-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so`。
- `python/sglang/srt/layers/quantization/fp8.py` 执行 `from vllm.plugins.dl_platform_plugin.ops.dl_fused_moe import ...` → 导入 `.venv` 的 vLLM → 加载 `.venv/.../vllm/_dl_C.so`。

两个*路径*（指向字节完全相同的 .so）→ `torch.ops.load_library` 把 `_dl_C` 注册了两次 → `c10::Error`。依赖导入顺序 → 间歇性（有时 `.venv` 的先加载，`hasattr(gemma_rms_norm)` 守卫短路了第二次加载）。

### 修复（已落地）
两个加载点现在都从**可导入的** vllm 推导路径（`import vllm; os.path.dirname(vllm.__file__)`），回退到旧的硬编码路径。单一路径 → 单次注册 → 确定性初始化。
- `python/sglang/srt/layers/layernorm.py`（`_dl_load_dl_C`）
- `python/sglang/srt/layers/quantization/fp8_utils.py`（`_ensure_dl_C`）

**注意：** 两个 `_dl_C.so` 字节完全相同（md5 `adc7f6a2…`），所以这是*正确性*修复，不影响速度。但它是让后续所有运行成为可能的关键解锁。

---

## 3. 已排除 —— "eager 拷贝 = 解码差距"（修正原假设）

tp4-gap 报告假设每解码步 ~21 次 `aten::copy_`/`to`（CPU 累计 ~290 ms）是差距来源。一个只读 agent 追踪了**整条解码热路径**（`TpModelWorker.forward_batch_generation` → `ForwardBatch.init_new` → `ModelRunner.forward` → `DecodeCudaGraphRunner.execute`/`load_batch`/`fill_from` → `sample`）：

**结论：拷贝数量解释不了 ~7 ms 的差距。** 每 token 的拷贝包括：
- **必需的** cuda-graph `fill_from`（≈6–7 次 `aten::copy_`/步——静态输入 buffer 的要求；这是机制，不是浪费）。
- 几个**很小的**标量/buffer 分配（`num_token_non_padded`、`clamp_position`、`mamba_track_mask`、overlap 的 `seq_lens+1`）——真实存在但都是亚毫秒级。
- **真正的延迟来源**是 **GPU 同步点**——尤其是 `SGLANG_DL_MULTI_STEP` setup 里的 `.item()` D2H 同步（`tp_worker.py:626-628`），加上 sglang 多进程 Engine scheduler 的**结构性 IPC 往返**（相对 vLLM）。

⇒ **消除拷贝收益很低。** 真正的杠杆是减少同步点 + IPC，而不是张量拷贝补丁。（完整的拷贝点排名清单已存入 agent 报告；最快的 quick-win 是预分配 `num_token_non_padded`/`positions` 标量——但每个都亚毫秒。）

---

## 4. 已定位 —— `SGLANG_DL_MULTI_STEP` KV pool 泄漏（那个有限的杠杆，未修）

tp4-gap 报告把 `SGLANG_DL_MULTI_STEP=4` 列为"~2 ms/token"的收益（每个 scheduler step 重放图 N 次，摊薄 host 开销）。`_dl_C` 修复后重测：崩溃。

### 现象
```
AssertionError: Unexpected overallocated KV cache, req.kv_committed_len=112, req.kv_allocated_len=208
```
把这个断言豁免（让超额分配的 free 路径跑起来）反而暴露出：
```
ValueError: pool memory leak detected! [full] total=568208, available=568000, evictable=64 ...
```
（144 个 slot 下落不明：`available + evictable + protected + session_held + uncached ≠ total`。）

### 根因（精确定位）
- `schedule_batch.py:2636`（`prepare_for_decode`，DL multi-step 分支）每步**投机性预分配 `_dl_n` 个 KV slot**：`out_cache_loc = alloc_for_decode(token_per_req=1)` + 循环 `_dl_n-1` 次 `alloc_for_decode`（临时推进 `seq_lens`）。`req.kv_allocated_len += _dl_n`；`req.kv_committed_len += 1`（多余的 `_dl_n-1` 由 `scheduler.py:3272/3366` 后续提交）。
- 当一个请求**在 multi-step batch 中途结束**（在用完所有 `_dl_n` 个投机分配的 slot 之前命中 `max_tokens`），超额分配 `[committed:allocated]` **没有被 `pop_overallocated_kv_cache` → `mem_cache/common.py:653` 回收**。`spec_algo is None` 的断言（`common.py:661`）挡住了非 spec 的回收路径；豁免它能让 free 跑起来，但记账仍然泄漏 144 个 slot → pool 不变量触发。
- 实测记账：`allocated = committed + output_len`（即输出 token 被有效计了两次），印证了投机分配与回收的不匹配。

### 为什么本次未修 → 已弃用（实测：multi-step 不是加速手段）
插桩并实测（两个泄漏检查都改为非致命以便跑完）：**multi-step 输出是正确的**（有效 JSON，与 CG 一��），但 **multi-step 慢约 4×，不是更快**——8.6 tok/s vs 基线 33.7。DL multi-step 循环里的 `.item()` D2H 同步（`tp_worker.py:626-628`，每步 3 次）加上投机分配/泄漏簿记占主导，淹没了任何 scheduler 往返的节省。**之前对 `SGLANG_DL_MULTI_STEP` "~2 ms/token" 的说法在 DLIN 上不成立**——得不偿失。⇒ **P0a″ 已弃用。** 其 KV 泄漏（真实的，边界对齐：`output % _dl_n == 0 → allocated` 翻倍）已无意义，因为 multi-step 反正更慢。修这个泄漏只会让一条慢路径稍微不那么漏。真正的杠杆是 **P0a′（在*正常*解码路径里减少同步点/IPC）**。

---

## 5. P0c —— sglang eager 乱码 JSON（仍未解决）

sglang FP8+**eager** 在 DLIN 上输出**乱码** JSON（`"name":"While the text mentions "...一个人的研究科学家`）；FP8+**cuda-graph** 正常。已确认它在 `_dl_C` 修复后**依然存在**（所以是个独立 bug，不是双重注册）。分歧在 `model_runner.py:924`（`disable_cuda_graph` 时 `decode_cuda_graph_runner = self.eager_runner`）——即 `eager_runner._forward_raw` 与捕获的图走了不同（在 DLIN 上有 bug）的 kernel 路径（可能是 Mamba/GDN 状态传递，或某个 CG 图替换掉的算子）。需要逐 token 的 logits 二分（eager vs CG）。

---

## 6. 本次交付的内容（真实、已验证）

| 项目 | 状态 |
|---|---|
| `_dl_C` 双重注册修复（`layernorm.py`、`fp8_utils.py`） | ✅ sglang 可确定性初始化 |
| `/dl-compare-sglang-vllm` 技能 → `.venv`，无 overlay，匹配 FP8 | ✅ |
| `scripts/dl/vllm_features_only.py`（独立的进程内 vLLM FP8+CG 评测脚本，带 main-guard） | ✅ |
| MRV1/MRV2 对比 | ✅ MRV1 崩溃 / MRV2 37.9 / sglang 33.7 |
| P0a 解码差距追踪 → "拷贝不是杠杆；同步点/IPC 才是" | ✅（重新定义了工作方向） |
| `SGLANG_DL_MULTI_STEP` KV 泄漏精确定位 | ✅（已定位，未修） |

---

## 7. 下一步优化计划（已修正、按优先级）

原计划里的 P0a（"消除 ~21 次 eager 拷贝"）已**弃用**——证明不是杠杆。修订如下：

### P0a′（现在的真正杠杆）—— 减少 GPU 同步点 + IPC
- **✅ 已交付（首个实测加速）：** `decode_cuda_graph_runner.py:624` **每个解码步**执行 `seq_lens.sum().item()`——一个 D2H 同步。改为 `int(seq_lens_cpu.sum())`（host 端计算，无同步；与已有的 `tbo_backend.py:195` 模式一致）。**实测：无约束解码 33.7 → 35.0 tok/s（+4%），JSON 输出有效（正确性保持）。** 差距：sglang 35.0 vs vLLM 37.9（从 33.7 在收窄）。
- **下一个同步点候选（已排除）：** `tp_active_ranks.detach().cpu().numpy()`（`model_runner.py:1711`）**仅 elastic-EP**（有门控，不在正常路径）；`handle.wait()`（`model_runner.py:2037`）是在线权重更新（一次性）。剩余延迟是 **scheduler↔worker IPC 往返**（结构性的）——需要 profiling（torch.profiler 在 DLIN 上不可靠；用 dlPTI）找更多每步同步点，再做架构层面的减少。

### ~~P0a″~~ —— `SGLANG_DL_MULTI_STEP` 已弃用（实测慢约 4×）
插桩 + 实测：multi-step 输出正确但 **8.6 tok/s vs 基线 33.7**（`.item()` 同步 + 投机分配簿记占主导）。在 DLIN 上不是加速手段。跳过。（KV 泄漏已刻画：`output % _dl_n == 0 → allocated` 翻倍；反正路径更慢，无意义。）

### P0b —— 在 cuda-graph 内启用 MoE 融合快速路径（`.enable_pdl` CG 阻塞）
`[[dlin-sglang-moe-pdl-cg-blocker]]`：sglang 在 CG 里被迫走慢的 `GEMMEX=2`，因为快的 `invoke_fused_moe_opt` 会崩（`per_token_group_quant_8bit_v2.cuh:396 .enable_pdl` 不能被 CG 捕获）。vLLM 用 CG + 同一个 `_dl_C.so` 却没问题 → 是 sglang 的捕获状态问题。修复 = 对齐 sglang 的 CG 捕获状态（stream/pool）。多天工作。

### P0c —— 根因 sglang eager 乱码 JSON
二分 `eager_runner._forward_raw` vs CG：对固定 prompt 在两者下 dump 每层 logits，找到第一个分歧点（很可能是 Mamba/GDN 状态传递，或某个 CG 替换的算子）。修复 → 解锁匹配 eager 的对比。

### P1a —— torch.compile 第二阶段（融合 norm/act+quant kernel）
注入 PostGrad pass manager，让 decode 用上 vLLM `_dl_C.so` 里已有的融合 kernel。有独立的计划文档（`docs/dl/torch-compile-stage0-1-plan.md`）。与 P0a′/P0a″/P0b 叠加。

### P2a/b —— 度量加固
重新设计前缀基准（长共享前缀 + 长输出 + 并发——RadixAttention 真正能体现的场景）；在 DeepSeek-V3 和 dense 模型上做广度测试。

---

## 8. 诚实的总结

sglang 的 DLIN 解码差距是**结构性 host 开销**（GPU kernel 完全一致）。本次：修了 `_dl_C` 初始化阻塞、**弃用 multi-step**（实测慢 4×）、并**交付了首个实测解码加速**——消除每步 `seq_lens.sum().item()` D2H 同步 → **sglang 33.7 → 35.0 tok/s（+4%）**，现在落后 vLLM 37.9 约 ~8%（之前 ~12%）。广泛扫描确认 greedy 解码路径里已没有 grep 可见的每步 `.item()` 同步点（其余都是 prefill/非 greedy/仅 elastic-EP）。剩余的 ~3 tok/s 是 **scheduler↔worker IPC 往返 + 更深的同步点**，只能通过 **dlPTI 解码步 profile** 看到（torch.profiler 在 DLIN 上不可靠）。这个，加上 P0b（MoE-CG）/ P1a（torch.compile 第二阶段）的叠加，是通向追平的多阶段路径。

### 落入代码树的改动（带 DL 标记）
- `python/sglang/srt/layers/layernorm.py`、`python/sglang/srt/layers/quantization/fp8_utils.py` —— `_dl_C` 单路径加载（修复双重注册 SIGABRT）。
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:624` —— `seq_lens_sum` host 端计算（消除每步 D2H 同步，+4% 解码）。
- 所有探索性 debug/绕过改动均已回退；代码树只保留上面两个真实修复。

---

## 复现

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh

# sglang FP8+CG（_dl_C 修复后）：
CUDA_VISIBLE_DEVICES=28,29,30,31 SKIP_VLLM=1 MEM_FRAC=0.6 \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py

# vLLM MRV2 FP8+CG：
CUDA_VISIBLE_DEVICES=28,29,30,31 VLLM_MODEL=/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ \
  VLLM_DTYPE=bfloat16 VLLM_EAGER=0 .venv/bin/python scripts/dl/vllm_features_only.py

# multi-step KV 泄漏复现（当前会崩——见 §4）：
CUDA_VISIBLE_DEVICES=28,29,30,31 SKIP_VLLM=1 SGLANG_DL_MULTI_STEP=4 \
  .venv/bin/python scripts/dl/bench_features_sglang_vllm.py
```

## 参考资料
- Memory：`dlin-sglang-vllm-dl_C-double-registration-fix`、`dlin-vllm-correct-package-and-env-gotchas`、`dlin-sglang-tp4-gpu-compute-gap`、`dlin-sglang-moe-pdl-cg-blocker`
- 仓库内：[`sglang-vs-vllm-features-json-prefix-dlin.md`](sglang-vs-vllm-features-json-prefix-dlin.md)、[`sglang-vs-vllm-tp4-20260715-report.md`](sglang-vs-vllm-tp4-20260715-report.md)、[`torch-compile-stage0-1-plan.md`](torch-compile-stage0-1-plan.md)
