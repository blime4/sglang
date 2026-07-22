# 完整支持 torch.compile —— 下一步战略 (2026-07-17)

> 目标定义："完整支持" = torch.compile 在 DLIN sglang 上 (1) 稳定跑通、(2) **有益**（编译后 ≤ eager，最好更快）、(3) 通用（跨模型）。
> 参考：`docs/dl/blog-sglang-dlin-qwen35-35b-tp4.md`（已据本周实测修正）。

## 当前状态（已达成 / 已证伪）

| 项 | 状态 | 证据 |
|---|---|---|
| compile 稳定跑通 + 输出正确 | ✅ 达成 | Stage 0 (`21970760c0`)；修了 3 个硬阻塞（gdc sitecustomize、track_mamba 抓图死锁、GDN recurrent 1.7GB functionalization-OOM） |
| wall_TPOT 与 vLLM 持平 | ✅ **已达成（靠 eager serving）** | sdk-dlop 下 sglang 26.53 ≈ vLLM 26.82ms（原"3ms 差距"是 sdk-0401 假象） |
| norm_quant 融合（dense FP8 linear） | ❌ 证伪 | DLIN `gptq_dlblas_gemmex` 内部量化，无 FP8 预量化 GEMM（`d793635541`） |
| act_quant 融合（MoE） | ❌ 证伪（no-op） | 移植+单元测试通过，但 GPU 实测 TPOT 无改善；MoE act-quant 在 `invoke_fused_moe_opt` 内部（`4785c99384`） |
| **编译后有益（compiled ≤ eager）** | ⚠️ **接近（mild catch-22）** | compiled decode GPU **22.13ms** vs eager 20.97ms（仅 +1.16ms，5.5%）。早期"80-101ms (3×)"是 `compile_check.py` 没设 serving-recipe env（无 `SGLANG_DL_GDN_DLIN` → 慢默认 GDN）的误测；正确配置下 dual compile+CG 两阶段都跑通+正确（深度验证 2026-07-17） |

## 核心阻塞：DLIN 融合算子的 catch-22

sglang 的 perf-critical 算子（GDN recurrent、MoE `invoke_fused_moe_opt`、FP8 `gptq_dlblas_gemmex`）都是 **DLIN 内部融合算子**。这造成不可调和的矛盾：

- **保持 opaque**（Phase-I 的选择，为避免分解膨胀）→ inductor 无法融合（norm+quant / act+quant 模式藏在算子内部）→ 无融合收益；**且** inductor 对周围"胶水"（norm/reshape/elementwise）生成的 kernel 比 sglang eager 的调优 kernel 更慢 → 编译后净变慢（80ms）。
- **放开分解**（vLLM 的路线）→ inductor 能融合，**但** (a) functionalization 会克隆 in-place 算子的状态（GDN ssm_states 1.7GB/层，已修；norm/MoE 较小）；(b) dense FP8 GEMM 一旦分解就被 inductor 换成自己的（更慢的）GEMM。

→ **纯 sglang 侧、靠"移植 inductor 融合 pass"无法让 torch.compile 变有益。** 这是本周 norm_quant + act_quant 两次证伪的共同根因。

vLLM 能做到，是因为它的 op 分解/注册策略不同（分解后用 cutlass/自家 fast kernel + 融合），不是单纯多几个 pass。

## 下一步路线（按工作量递增）

### 路线 1：接受现状 + 收尾（推荐作为当前里程碑）
- Stage 0 已落地（compile 跑通+正确）；wall_TPOT 已持平 vLLM（靠 eager + sdk-dlop）。
- torch.compile 当前状态：**"能跑但不益"**（80ms）。对生产用 eager serving；compile 留作未来/其它模型的基座。
- 工作量：仅文档/记忆收尾。**性价比最高**（目标"持平 vLLM"已达成）。

### 路线 2：分解+融合（vLLM 等价路线，多日，纯 sglang 侧）
翻转 opaque 策略：让 inductor 分解 DLIN 算子 → 融合 norm/act/rope。
- **第一步（可行性门）**：枚举分解后的 functionalization 克隆源（哪些 in-place 算子克隆、多大）。已知：GDN ssm_states（1.7GB，已 @torch.compiler.disable）。待查：`gemma_rms_norm`/`fused_add_gemma_rms_norm`（in-place，激活大小，小）、`invoke_fused_moe_opt`（mutate topk/sorted_ids，小）。
- **致命点**：dense FP8 GEMM 分解 → inductor 慢 GEMM。除非 DLIN 算子能在"分解态"下仍调用 fast 内核（需 DLIN 配合）。
- 结论：**可行性取决于 DLIN 算子能否在分解态保持 fast** —— 这超出了"纯移植 pass"，半只脚踏入路线 3。

### 路线 3：DLIN-aware inductor 后端（重大，最通用）
把 DLIN 调优 kernel 注册为 inductor 的 extern lib / 自定义后端 → 编译图 = DLIN-fast kernel + inductor 融合。
- 最彻底地解决 catch-22（fast kernel + 融合兼得），且通用（跨模型）。
- 工作量最大（需 DLIN 编译器/算子团队配合，或深度 inductor 改造）。

### 路线 4（外部）：请 DLIN 出"融合友好"算子
让 DLIN 把 norm+quant / act+quant 做成**接受预量化输入的独立融合 op**（而非内部融合）→ sglang 直接调 + inductor 可组合。
- 用户本轮约束是"纯 sglang 侧"，此路线被排除；但若放宽，这是最小 sglang 改动路径。

## 决策门
- 若目标仅是"持平 vLLM 延迟" → **路线 1**，已达成，收尾。
- 若目标是"torch.compile 真正有益（GPU 也超 vLLM）" → 需 **路线 2 的可行性门**先过（枚举克隆 + 验证分解态 dense GEMM 是否可接受），不过则转 **路线 3/4**。
- 关键未知：**DLIN 融合算子在"分解态"能否保持 fast**。这是整个"有益 compile"的分水岭，建议作为下一个调研的首要问题。

## 已落地资产（无论走哪条路都保留）
- Stage 0 代码（`21970760c0`）：compile 跑通 + 正确，gated/DL-marked，eager serving 不受影响。
- Task 5 注入 + crash 修复（`d81dd520e3`/`2a69523c50`）：PostGradPassManager 接进 decode compile。
- act_quant pass（`4785c99384`）：unit-tested、gated，作"已验证非收益"留档。
- `VLLM_DL_TIME_REPLAY` hook（overlay `cuda_graph.py:360-389`）：vLLM GPU-forward 直接实测能力。
- 修正后的 blog + 本计划 + 记忆：完整可复现的真相链。
