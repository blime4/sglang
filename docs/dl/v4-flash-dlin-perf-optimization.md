# V4-Flash DLIN 性能优化（2026-07-27）

**前提**：V4-Flash 已产出**正确**输出（3 个 bug 修复：MLA `causal`、`hc_post` no-op、MoE FP4 `use_mxfp4_w4a16`）。本文聚焦**吞吐性能**（decode 1.6 tok/s → 目标提速）。

## 当前性能（eager, TP8, 32GB×8, cards 24-31）

- **decode: 1.60 tok/s（TPOT 761ms）**；short-prefill(5tok) 1.16s；long-prefill(500tok) 超时（>300s）。
- 期望（按 149GB/TP8 memory-bound floor）~6 tok/s → 有 ~4× 缺口，**几乎全在 host 开销**。

## Phase 0/1 — per-component 拆解（CUDA-event 计时, `SGLANG_DL_TIMING=1`）

每 decode step（43 层累计）：

| 成分 | 时间 | 占比 | 性质 |
|---|---|---|---|
| **MLA 投影（wq_a/wq_b/wkv/wo FP8 GEMV）+ rope + norm** | ~407ms | 53% | host 开销主导（见下） |
| MHC-mix + host（hc_pre/hc_post torch fallback + launch） | 157ms | 21% | torch fallback |
| MoE（FP4 `use_mxfp4_w4a16`） | 37ms | 5% | **快** |
| `flash_mla`（DL attention op） | 18ms | 2% | **快** |
| indexer | 可忽略 | ~0% | per-batch |

**DL op 本身都很快**（flash_mla 18ms + MoE 37ms = 55ms 真实 GPU 计算）。瓶颈是**投影的 host 开销**，不是 op 计算。所有投影是 **FP8 (F8_E4M3)**（已量化，排除 bf16 假设）。

## Phase 2 — dlPTI 证实（host launch/patch overhead）

`dlpti_tools capture --activity-mask cu`（CUDA runtime），每 GPU/4-token decode 窗口（~3s）：

| Runtime call（host 端） | 时间 | 含义 |
|---|---|---|
| **`cuLaunchKernel`** | **1072ms** | ~27,500 次 kernel launch × ~40µs 排队 |
| **`dlcuGraphExecNodeApplyPatch_`** | **408ms** | DL 运行时内部 CUDA-graph 每步打补丁（更新参数） |
| `dlcuGraphLaunchMultiInstance_` | 36ms | graph launch |
| `cuMemcpyDtoDAsync` / `cuEventRecord` / ... | ~30ms | 杂项 |

`cmd`（device command）导出仅 9.8KB → 证实 **GPU 工作在 CUDA graphs 内部**（DL 运行时已 graph 化部分 op），顶层 device 命令很少。

**结论：瓶颈 = 主机端 CUDA-runtime 串行开销。** 每 step 每卡 ~110,000 个 runtime 调用（launch + graph 补丁 + memcpy + event），每个 ~14µs，总计 ~400ms/token 的主机时间。**GPU 在等主机排队下一个 kernel**——这是 "GPU 空转等 launch" 的根因，~50% decode 时间是 host overhead。

## 优化路径（按杠杆排序）

### 1. sglang 级 CG（最大杠杆，硬件受限）
把整个 decode step 捕获成一个 graph：27,500 次 launch + 多次 per-op 补丁 → **1 次 graph launch + 1 次补丁**。预估 TPOT 761ms → ~300-400ms（消除 ~370ms host 开销，~2× 提速）。
- **硬件受限**：149GB 模型 + CG workspace 放不进 32GB 卡（`mem_fraction_static` 0.75–0.90 都 OOM/hang）。
- 需要 48GB+ 卡，或先压 KV/activation 占用。

### 2. 减少 launch 数（fusion，sglang-Python 侧，不依赖 CG）
投影/小 op 被拆成大量小 launch；fuse wqkv/q_norm/rope/store/wo，减少 launch 数。受限于 DL op 内部 kernel 拆分（部分 fragmentation 在 DL runtime，sglang 侧 fuse 空间有限）。

### 3. 减少 graph 补丁开销
调查 `dlcuGraphExecNodeApplyPatch_`（102ms/token）为什么 per-op graph 每步都要 patch；sglang CG 会把这变成 1 次整图补丁。

### 不做（已证伪）
- **DL op 优化**：flash_mla/MoE 已经很快（55ms），不是杠杆。
- **indexer → DL op**：indexer 可忽略。
- **投影 bf16 量化**：投影已是 FP8。

## 复现命令
```bash
source sdk-dlop-07-13-20-30/env.sh
export CUDA_VISIBLE_DEVICES=24,25,26,27,28,29,30,31  # 或其他空闲 TP8 块
# V4 env（FP4 MoE 自动检测，无需 SGLANG_DL_MOE_FP4）:
export DLI_V2=ON TORCHDYNAMO_DISABLE=1 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
       SGLANG_DL_MOE_FUSED=1 SGLANG_DL_FP8_Q2=1 SGLANG_DL_MOE_FUSED_MAX_M=2048 \
       SGLANG_DL_GDN_DLIN=1 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 \
       SGLANG_OPT_USE_FUSED_HASH_TOPK=0 SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0 \
       SGLANG_OPT_USE_TOPK_V2=0 SGLANG_TOPK_TRANSFORM_512_TORCH=1 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# 吞吐：/tmp/v4_bench.py（decode TPOT + prefill）
# dlPTI: dlpti_tools capture --activity-mask cu --data-file cap.db -- .venv/bin/python <smoke>
#   export --export-range 95%:100% --format perfetto-json cap.db; ijson 解析（见 /tmp/parse_cap.py）
```

## 已提交的相关 commits
- `e5b26b5049` MoE FP4 `use_mxfp4_w4a16`（gibberish root cause）
- `2d1d8e4aa3` hc_post no-op 修复
- `29d8b28e35` MLA causal 修复
- `719879c564` per-layer stats 诊断工具（`SGLANG_DL_LAYER_STATS=1`）
