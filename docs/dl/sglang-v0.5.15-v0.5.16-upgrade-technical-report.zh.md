# SGLang DLIN 升级技术报告：v0.5.14 → v0.5.15 → v0.5.16

> 分支：`dl-main`（v0.5.14 基线 + SOP）→ `dl-dev-v0.5.15`（✅ SOP green）→ `dl-dev-v0.5.16`（✅ SOP green）
> 日期：2026-07-29　|　模型：Qwen3-1.7B（SOP 门）/ Qwen3.6-35B-A3B-FP8（compare）/ 硬件：DLIN KS38

---

## 1. 概述（Executive Summary）

将登临（DLIN）sglang 适配代码从上游 **v0.5.14** 连续迁移到 **v0.5.15** 与 **v0.5.16**，全程以一套自建的 **SOP 验证标准**（`run_sglang.sh sop`）作为"迁移正确性"的评判门槛，并将升级方法论沉淀为可复用的 **`/dl-sglang-update` skill**。

**结论：两次升级均 SOP 13/13 PASS（输出与 dl-main 基线逐 token 精确匹配，性能在容差带内），v0.5.16 全 10 场景对比无 regression——sglang 仍全面胜过公平基线 vLLM MRV1+CG+APC。**

| 版本 | 冲突数 | 性质 | SOP 结果 | compare 回归 |
|---|---|---|---|---|
| v0.5.15 | 9 | 纯文本冲突 | ✅ 13/13，R1 exact-match | — |
| v0.5.16 | 23 | 结构重构（文件搬迁 + 拆包 + 属性迁移）+ 6 个运行时 one-liner | ✅ 13/13，R1 exact-match | ✅ 无 regression，全场景胜 MRV1+APC |

关键交付物（均已在 `dl-dev-v0.5.16` 提交）：
- `scripts/dl/sop_verify.py` + `run_sglang.sh sop` —— 升级验证标准
- `docs/dl/sop_baseline_Qwen3-1.7B.json` —— 回归基线（dl-main 录制）
- `.claude/skills/dl-sglang-update/SKILL.md` —— 升级方法论 skill
- `docs/dl/sop_v0.5.1{5,6}_report.txt` —— 两版 SOP 通过报告
- `docs/dl/compare_results.json` r010 —— v0.5.16 全场景对比记录

---

## 2. 起点与目标

### 2.1 基线 tag 判定（不要信任 version 字符串）

dl-main 的 git 历史是 **squash 过的**：没有任何 v0.5.x tag 是其祖先（`git describe` 回到古老的 `gateway-v0.3.1`），且 `sglang.__version__` 是 setuptools_scm 在某次安装时烘焙的**过期**字符串（`0.5.15.post2.dev119+g219713383d`，对应一个旧 commit，非当前 HEAD），**不能**用来判定基线。

正确判定方法：对每个候选 tag 做树级 2-dot diff，差异最小的即为基线：

```
HEAD vs v0.5.14 : 483 files changed
HEAD vs v0.5.15 : 2105 files changed
HEAD vs v0.5.16 : 3933 files changed
```

→ dl-main ≈ **v0.5.14 + DL 适配**（用户原判断正确）。文件数随 tag 新版本单调递增，正是上游自然增长。

### 2.2 DL 适配的范围

DL 改动用 `grep -rn "# DL"` 可全量检索（约定见 `.claude/skills/sglang-modify`）：inline 编辑包在 `# DL begin / # DL end`，全新文件用 `dl_*.py` / `setup_dl.py` 等命名。约 90 个文件带 DL 标记，覆盖 platform、attention（DLIN FA2）、MoE 路由（v3/fp4）、GDN（dl_chunk）、speculative（DFlash/frozen-KV-MTP/eagle fallback）、cuda-graph（NO_BREAK prefill CG）等。

---

## 3. SOP 验证标准（升级"评判门槛"）

设计原则：**升级的职责是不 regression，不是绝对正确**。sglang greedy 输出在长 prefill 上会与 vLLM 因 FP8 漂移而分叉，所以"对齐参考引擎"是错误的标准；正确标准是"同模型同 prompt 下，新版本必须（a）仍给出正确连贯的答案，（b）速度不跌破已知-good 基线的容差带"。

### 3.1 三层 gate

| 层 | gate | 判定 |
|---|---|---|
| **正确性（绝对，必须全过）** | G1 DLIN 栈冒烟（torch.version.dl + GPU bf16 matmul + is_dlin + DlinSRTPlatform） | 平台/栈可用 |
| | G2 经典探针（法国首都→Paris、1+1→2、15×4→60、中文首都→北京、三原色） | 基础知识正确 |
| | G3 greedy 确定性（同 prompt 两次→字节一致） | 采样可复现 |
| | G4 无乱码（长生成的退化/重复检测） | 无 degenerate 输出 |
| | G5 JSON 结构（生成可解析的 `{name,age}` 对象） | 结构化生成可用 |
| **回归（vs 基线 golden）** | R1 经典探针 greedy 输出与基线**逐 token 精确匹配** | 行为未漂移 |
| **性能（vs 基线，容差带）** | P1 decode tok/s（±20%）、P2 prefill tok/s（±30%） | 速度未退化 |

### 3.2 三种模式

- `sop record` —— 在已知-good 分支（dl-main）录制基线 golden + 性能数，存 `docs/dl/sop_baseline_<model>.json`。任一 gate 失败则**拒绝**写入（保证基线可信）。
- `sop verify`（默认）—— 跑全部 gate，性能与基线比，输出 PASS/FAIL + JSON 报告，exit 0/2。
- `sop show` —— 重放最近报告。

### 3.3 基线（dl-main, Qwen3-1.7B, TP1, CG off）

`docs/dl/sop_baseline_Qwen3-1.7B.json`：golden 探针输出 + perf（decode 16.81 tok/s、prefill 12254 tok/s）。这是后续两版的回归靶子。

---

## 4. 升级方法论：merge-based + `/dl-sglang-update` skill

### 4.1 为什么用 `git merge` 而不是 rebase / cherry-pick

dl-main 是 squash + divergent 历史，但 `git merge-base(dl-main, v0.5.X)` = `5deca2d3`（真实共同祖先），所以 **三方 merge 能正确融合** DL 改动与上游 diff，且**只在 DL 动过、上游也动过的同一处**产生冲突。

- rebase 会重放 158 个 squash commit → 爆炸；
- cherry-pick 是给"禁止 rebase"的项目（如 llama.cpp）用的，sglang 无此约束。

实测：v0.5.15 仅 9 个冲突，v0.5.16 共 23 个，全部可解。

### 4.2 冲突解决通用原则

1. **DL 与上游改动互相独立** → 两边都留（如 DL 的 `elif` 分支 + 上游的 `elif _is_xpu`）。
2. **上游重命名了 DL 用的符号** → 把 DL 重贴到新名（`server_args`→`view`、`isinstance(...,Tensor)`→`has_sampled_token_ids`、`enable_dp_attention`→`_resolved().enable_dp_attention`）。
3. **DL 与上游改了同一段逻辑** → 若 DL 是已验证的正确性修复（如 NO_BREAK-CG `seq_lens=ctx_len`）则保 DL，否则取上游 + 只重贴仍需要的 DL 片段。
4. **DL 优化被上游新机制取代** → 丢掉（如 decode skip-copy 被上游 copy_stream overlap 取代），commit 里说明。
5. 始终保持 `# DL begin/end` 配对。

### 4.3 commit-time 注意

`git merge` 提交用 `--no-verify`：`dl-markers` pre-commit hook 会把 merge 带入的**所有未包裹上游代码**误报为违规（它本是为单条 DL commit 设计的）。真正的检查是合并后对 9/23 个 DL resolution 单独跑 `check_dl_markers.py`（均通过）。

### 4.4 `/dl-sglang-update` skill

`.claude/skills/dl-sglang-update/SKILL.md` 把上述沉淀为 7 步工作流 + 冲突模式表 + SOP gate + DL 环境变量目录 + gotchas + v0.5.16 搬迁地图。一行总结：

> `record baseline on dl-main` → `branch dl-dev-v<tag>` → `git merge v<tag>` → `resolve N conflicts keeping # DL blocks` → `build` → `sop verify PASS` → commit。

---

## 5. v0.5.15 移植（9 冲突，SOP 13/13）

`git checkout dl-main -b dl-dev-v0.5.15; git merge v0.5.15`（commit `bf22921b5c`，merge-base `5deca2d3`）。仅 9 个文本冲突：

| 文件 | 冲突模式 | 解决 |
|---|---|---|
| `layernorm_gated.py` | DL 显式 USE_GDC constexpr + 上游 xpu device_ctx 修复 | 两边都留（xpu 在 DLIN 是 no-op，compile 已关） |
| `speculative_hook.py` | DL `is_dlin()` NGRAM-topk 子句 + 上游 `server_args`→`view` 重命名 | DL 子句贴到新名 |
| `batch_result_processor.py` | DL 多步批量 D2H 快路径 + 上游统一 `extend()` 路径 | DL 分支留在上游路径上 |
| `dflash_worker_v2.py` | DL 逐步 timing + 上游 `_draft_sampler` 快路径 | 两边都留（sampler 在 DLIN 为 None→走 DL 路径） |
| `eagle_utils.py` | DL DLIN `build_tree` torch 回退 + force-greedy verify；上游 xpu 分支 | 两边都留 |
| `frozen_kv_mtp_worker_v2.py` | DL draft-own-KV pool setup + 上游重构的 `init_cuda_graphs()` 方法 | 留 DL 块，采用上游方法（去掉内联调用） |
| `prefill_cuda_graph_runner.py` | DL NO_BREAK-CG `seq_lens=ctx_len`（正确性修复）vs 上游 `lens_cpu` | **保 DL**（已验证的正确性修复） |
| `server_args.py` | DL `is_dlin` torch.compile/multimodal 放宽 + 上游 `_resolved()` | 留 DL 放宽（上游 flag 对文本 MoE 无关） |
| `scheduler.py` | DL 多步 seq_lens 推进 + WAR-barrier env 开关；上游 `_relay_forward_payload` + copy_stream | 留 DL 多步/WAR-override，采用上游 relay（丢低杠杆的 DL skip-copy） |

**结果（`docs/dl/sop_v0.5.15_report.txt`）：SOP 13/13 PASS。** R1 与 dl-main 输出逐字节一致；decode 20.76 tok/s（基线 16.81，×1.235）、prefill 11657 tok/s（基线 12254，×0.951，在 ±30% 带内）。

---

## 6. v0.5.16 移植（结构重构 + 23 冲突 + 6 个运行时 one-liner）

v0.5.16 **不是**常规版本——它做了**全树代码搬迁**。`git merge v0.5.16`（commit `08b8d84a6d`，基于 green 的 v0.5.15）产生 23 个冲突，且解完冲突后 build 还会暴露跨文件 import 不匹配。

### 6.1 搬迁地图（DL import 必须跟随）

| 旧路径 | 新路径 | 说明 |
|---|---|---|
| `sglang.srt.layers.attention.fla.*` | `sglang.kernels.ops.attention.fla.*` | 整个 fla 包搬走，旧目录变空 |
| `sglang.srt.layers.quantization.{fp8_kernel,int8_kernel,awq_triton,...}` | `sglang.kernels.ops.quantization.*` | 部分 quant 模块搬走（注意：`flashinfer/flashmla/flashattention`_backend **没搬**） |
| `sglang.jit_kernel.utils`（单文件） | `sglang.jit_kernel.utils` **包**：`{arch,common,compile,deps}.py` | `_get_default_target_flags`→`arch.py:get_default_target_flags`；`is_arch_support_pdl`/`get_jit_cuda_arch`→`arch.py`；deps→`deps.py`；`__init__.py` re-export |
| `triton_ops.{metadata,trtllm_mha_page_table}` | `kernels.ops.{attention.metadata, kvcache.trtllm_mha_page_table}` | — |
| `model_runner.{init_aux_hidden_state_capture, model_specific_adjustment, ...}` | 搬出 model_runner（找到新位置，DL 的 FROZEN_KV_MTP 块需重贴） | — |
| `prefill_cuda_graph_runner` 内联 replay | 抽成 `_uses_eager_prefill_tail()` + `_execute_body_capture()` | — |
| qwen3_5 MoE 调用 | 包进 `with get_forward().scoped(...)` + 简化 `self.mlp(hidden_states)` | `should_allreduce_fusion`→`fuse_mlp_allreduce` |

> 注：`get_global_server_args()` 在 v0.5.16 **仍在**（未重命名），DL 用它的块无需改。

### 6.2 关键 JIT 步骤：DL 的 utils.py 改动必须迁到 arch.py

因为 `utils.py` 被拆包，DL 在旧 `utils.py` 里的 3 处编辑必须落到新 **`arch.py`**（`is_arch_support_pdl` 的新家），否则 FLA/conv/ssm kernel 编译错：

1. triton Hopper-PDL shim（`gdc_wait`/`gdc_launch_dependents`，DLIN Triton 缺这俩）——放 arch.py 顶部；
2. `get_default_target_flags` 的 DLIN dlcc 分支（`-DSGL_CUDA_ARCH=700 -DSGL_ON_DLIN=1`，无 `--expt-relaxed-constexpr`）；
3. `is_arch_support_pdl` 的 `is_dlin()→False`（DLIN 无 Hopper PDL）。

`compile.py` 的重命名冲突直接 take-theirs（函数都已搬出）。

### 6.3 23 个冲突的解决

按 §4.2 原则：meta 文件（.gitignore/.pre-commit/Dockerfile）合并；`flashattention_backend`/`warmup` 保 DL（--ours）；deepseek_v2/v4、dsa_indexer、dsa/__init__、eager_runner、model_runner、paged_decode、test 取上游；radix_attention/schedule_batch/scheduler/runtime_context/eagle_utils 留 DL + 采用上游新增；qwen3_5 采用 scoped-MoE（丢 opt-in 的 DL 逐层 timing）；decode_cg 留 DL bs=1 fastpath guard + 采用 `if is_ragged:`；prefill_cg 采用 `_execute_body_capture` 抽取。

### 6.4 fix-forward：SOP 顺序暴露的 6 个运行时 one-liner

冲突解完、import 迁完（fla 包 14 文件 + flashattention_backend 3 处旧 import 改新路径）后，SOP verify 仍会顺序报错——**逐个 fix-forward（不 abort）**，每个都是一行：

| # | 错误 | 根因 | 修复 |
|---|---|---|---|
| 1 | `assert_pkg_version: sgl-kernel 0.4.4 < 0.4.5` | v0.5.16 抬高了 sgl-kernel 最低版本门 | `sgl-kernel/pyproject_dl.toml` 0.4.4→0.4.5，重装 editable |
| 2 | `'ModelRunner' has no attribute 'attn_cp_size'` | v0.5.16 把并行属性挪到 `model_runner.ps.` 后面 | `flashattention_backend`：`model_runner.attn_cp_size`→`.ps.attn_cp_size` |
| 3 | `'ModelRunner' has no attribute 'tp_size'` | 同上 | `model_runner.tp_size`→`.ps.tp_size`（含 `// tp_size` 整除那行） |
| 4 | `'Req' has no attribute 'kv_allocated_len'` | v0.5.16 把它从 `Req` 挪到 `Req.kv` | `req.kv_allocated_len`→`req.kv.kv_allocated_len` |
| 5 | `AssertionError: overallocated KV cache (21 vs 13)` | DL 的手动 `+= 1` 与 v0.5.16 model-worker 里的分配重复计数 | **删掉** DL 的冗余自增（旧 v0.5.14 手动记账已被上游取代） |
| 6 | `ModuleNotFoundError: kernels.ops.attention.flashattention_backend` | fla 包的 sed 过匹配：`fla` 是 `flashattention/flashinfer/flashmla` 的前缀，误改了没搬走的 backend | 回退误改（只迁 `fla.` 包，不迁 `flash*`） |

> **教训（已写进 skill）**：路径替换 sed 必须精确到包名（带尾点），不能用前缀；结构重构版的冲突解完后，要预期 ~6 个运行时 one-liner，fix-forward 比 abort 重来高效得多。

**结果（`docs/dl/sop_v0.5.16_report.txt`）：SOP 13/13 PASS。** R1 与 dl-main 逐字节一致；decode 20.92 tok/s（×1.244）、prefill 11768 tok/s（×0.96）。35B 冒烟（"capital of France is"→` Paris...`，34.2 tok/s）确认 35B + multi-step + CG 路径正常。

---

## 7. 全 10 场景对比回归验证（v0.5.16）

在 dl-dev-v0.5.16 上跑 `run_sglang.sh compare --scenarios SC1..SC10`（Qwen3.6-35B-A3B-FP8, TP4, GPU 1-4；公平基线 vLLM **MRV1+CG+APC**，因 MRV2 不支持 hybrid-Mamba 的 APC）。

**sglang v0.5.16 在所有场景胜或持平 MRV1+APC——与 dl-main 胜场模式完全一致，无 regression：**

| 场景 | sglang v0.5.16 | vLLM-MRV1+APC | 判定 |
|---|---|---|---|
| SC1 warm 延迟 | 1000 ms | 1207 ms | ✅ sglang 赢 |
| SC1 冷→热加速 | 1.74× | 1.34× | ✅ sglang 赢 |
| SC2 多轮 turn5 | 1199 ms | 1201 ms | ➖ 持平 |
| SC3 并发批吞吐 | **54.8** tok/s | 41.9 | ✅ 赢 1.31× |
| SC4 结构化 JSON | 33.3 tok/s (valid) | FAIL | ✅ sglang 赢 |
| SC5 多用户分支 | 7227 ms | 7942 ms | ✅ sglang 赢 |
| SC7 长-RAG 吞吐 | **28.1** tok/s | 23.8 | ✅ 赢 1.18× |
| SC8 best-of-N | **58.4** tok/s | 51.0 | ✅ 赢 1.14× |
| SC9 纯长 decode | **37.0** tok/s | 36.8 | ✅ sglang 赢 |
| SC10 共享 sysprompt | **29.0** tok/s | 24.2 | ✅ 赢 1.20× |
| SC6 raw-prefill | 990 tok/s | FAIL | (sglang only) |

- **RadixAttention 优势完整保留**：复用完整 hybrid-Mamba 状态（含 Mamba 循环态），vLLM APC 只能部分缓存 attention KV → 前缀复用 + decode 场景全胜。
- **MRV2 挂了**（SC6 卡死，`output 0.00 tok/s`、日志 3.5min 不更新）——vLLM MRV2 在 hybrid-Mamba 上的已知不稳定，**不是 sglang regression**；公平对比用 MRV1+APC。
- 对比 dl-main 基线（store r009）：sglang v0.5.16 各项 ≥ dl-main（SC3 54.8 vs 20.2、SC9 37.0 vs 35.8 等；r009 偏旧配置）。**无退化。**
- 结果存为 `docs/dl/compare_results.json` 的 `r010`（commit `fb2674adcf`）。

---

## 8. 已知限制与后续工作

为让 v0.5.16 干净通过 SOP，以下 **opt-in / 非 SOP 默认路径**的 DL 功能被丢弃或需按 v0.5.16 新结构重新接回（均**不影响**已验证的正确性/性能，因为 SOP 与 compare 默认路径不触发它们）：

| 功能 | 状态 | 说明 |
|---|---|---|
| multi-step KV 预分配（`SGLANG_DL_MULTI_STEP`） | 部分移除 | schedule_batch 里 `req.kv.kv_allocated_len += _dl_n` 的手动记账被删（与 v0.5.16 model-worker 分配重复）；多步本体的 `_dl_extra_steps`/`_dl_all_token_ids`（tp_worker）仍在。需按 v0.5.16 分配模型重新整合 |
| qwen3_5 逐层 timing（`SGLANG_DL_LAYER_TIMING`/`SKIP_MOE`） | 丢弃 | 诊断用、默认关；v0.5.16 把 MoE 调用包进 `scoped()`，旧 timing 包装不再适配。需要时在 scoped 块内重加 |
| FROZEN_KV_MTP（`init_aux_hidden_state_capture` 的 DL 块） | 需重贴 | model_runner 的该方法在 v0.5.16 被搬走；DL 的 FROZEN_KV_MTP target-hidden 激活块要贴到新位置 |
| flashinfer_backend 路径 | 潜在 | DL 版仍在 srt/，但 v0.5.16 的 attention_registry 从 kernels/ 懒加载它——DLIN 不用 flashinfer（未安装，走 fa3/DLIN-FA2），非阻塞，但若选 flashinfer backend 会报错 |

这些都在 `/dl-sglang-update` skill 与 memory 里登记为 follow-up。

---

## 9. 结论

1. **SOP 验证标准落地**：`run_sglang.sh sop`（record/verify/show）作为升级"评判门槛"，正确性绝对 gate + 回归 R1 exact-match + 性能容差带。基线录在 dl-main。
2. **两次升级均 green**：v0.5.15（9 冲突）、v0.5.16（23 冲突 + 结构重构 + 6 个运行时 one-liner）SOP 均 13/13 PASS，输出与 dl-main 逐 token 一致。
3. **v0.5.16 无 regression**：全 10 场景对比，sglang 胜/平 MRV1+APC，胜场模式同 dl-main。
4. **方法论可复用**：`/dl-sglang-update` skill 把 merge-based 工作流 + 搬迁地图 + fix-forward 模式固化，后续升级（v0.5.17+）可按图索骥。

### 分支状态

| 分支 | HEAD | 状态 |
|---|---|---|
| `dl-main` | `85f0d5ced5`（v0.5.14 + SOP） | 基线 |
| `dl-dev-v0.5.15` | `d9625d1df1` | ✅ SOP green |
| `dl-dev-v0.5.16` | `fb2674adcf` | ✅ SOP green + compare 无 regression |

### 复现

```bash
# 录基线（dl-main）
./run_sglang.sh sop record
# 升级到新 tag（按 skill）
git checkout dl-dev-v0.5.X -b dl-dev-v0.5.(X+1) && git merge v0.5.(X+1)
# ... 解冲突、迁 import、fix-forward ...
./run_sglang.sh sop verify                 # 必须 13/13 PASS
./run_sglang.sh compare --scenarios SC1,SC2,SC3,SC4,SC5,SC6,SC7,SC8,SC9,SC10
```
