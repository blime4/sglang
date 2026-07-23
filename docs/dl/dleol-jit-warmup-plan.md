# DLIN dleol JIT 全面预热计划 (vLLM-style: 服务期零 JIT)

目标:在 serve 启动 warmup 阶段完成**所有** dleol/triton JIT 预编译,使真实请求
**零首命中 JIT 尖峰**,达到 vLLM AOT `_dl_C.so` 的效果(vLLM 运行时零 JIT)。

## 设计核心:warmup 按_capture size list_进行(对齐 DLIN vLLM)

DLIN vLLM 的做法:启动时按 **capture size list** 预热 ——
`vllm/v1/worker/gpu_worker.py:575` 构造 `warmup_sizes = compile_sizes + cg_capture_sizes`,
`cudagraph_utils.py:134` `for num_tokens in capture_sizes` 逐大小跑前向 → 触发对应形状的
JIT 编译。

sglang 的对应物就是 `server_args.cuda_graph_config`:
- `prefill.bs` = `[4,8,12,...,2048]`(预填充 capture/compile 大小)
- `decode.bs` = `[1,2]`(解码 capture 大小)

→ warmup 读取这两份 capture list,逐大小发前向,把 fused-MoE 等 per-shape kernel
对这些**引擎实际会服务的尺寸**全部预编译。这就是 `-W/--dl-warmup` 的
`dlin_capture_sizes` warmup(`python/sglang/srt/entrypoints/warmup.py`)。

## 现状(2026-07-24)

- **已实现** `dlin_capture_sizes` warmup:读 `cuda_graph_config.{prefill,decode}.bs`,
  dedup 后逐大小 JIT 预热;`SGLANG_DL_WARMUP_SHAPES` 可覆盖为子集。`-W` 开启。
- **fused MoE `invoke_fused_moe_opt`**:per-M-shape JIT,~20-85s/shape 首命中,重复 ~1-2s,
  缓存于 `~/.triton/cache`(`DLEOL_CACHE_SIZE`)。
- **fa3 / DL flash attention**:已 AOT(`_sgl_fa2_C`/`_dl_C.so`),**无 JIT**。
- decode capture sizes(`decode.bs`)已被 cuda-graph capture 在启动时预热;warmup 再加固。
- prefill capture sizes:prefill CG 在 DLIN 是 `disabled`,capture 不会预热它们 →
  warmup 是覆盖这条 gap 的主力。

## Phase 0 — 盘点确认 JIT 面(先做,1 次 profile)

用 `dl-profile` / jit_monitor 跑冷启动 serve + 代表性负载,确认:
- 除了 fused MoE,还有哪些 per-shape kernel JIT(fused_qk_norm / mamba conv1d / rope / …)。
- `dlin_capture_sizes` 走 input_ids+max_new_tokens=1 的 prefill,是否覆盖到这些 kernel;
  若有遗漏(如 ~10s 残留来自 attention 的某个 triton op),把 warmup 请求改成**全路径**
  (text prompt + 足够 decode token)以覆盖。

成功判据:真实请求期 jit_monitor 零 JIT 事件。

### Phase 0 结果(2026-07-24,已实测)— capture-list warmup 对 prefill **不够**

`serve -W`(44 个 capture size 全量预热完成)后,真实 text 请求计时:

| 请求 | prompt_tok | 首命中 | 重复 |
|---|---|---|---|
| short | 144 | 3.52s | 0.59s |
| mid(非 capture) | **929** | **18.01s JIT** | 2.35s |
| long(非 capture) | **1858** | **22.26s JIT** | 0.69s |

**结论:sglang 的 eager prefill 产生任意 M,不会 snap 到 capture size**(929 落在 capture
896/960 之间、1858 落在 1792/2048 之间 → 都没被预热 → 仍 JIT ~18-22s)。capture-list warmup
只覆盖**恰好等于 capture size 的 prefill**(罕见)+ decode(capture-sized batch)。
**vLLM 不靠 per-shape warmup 消除 JIT —— 它是 AOT `_dl_C.so`,运行时零 JIT**;它的 capture-size
warmup 是给 cuda-graph capture 用的,不是给 JIT 用的。所以 "vLLM-style capture-list warmup"
在 sglang(DLIN,有 per-M JIT)上**无法达到零运行时 JIT**。

→ 要真正零 JIT,只能:**Phase 3(M-agnostic kernel,根因解)**,或密集预热每个 M
(类 voice_chat step=4,但启动极慢 + cache ship),或让 sglang 把 prefill M round 到
capture size(开 prefill cuda-graph / 加 prefill bucketing —— 但 DLIN 上 prefill CG 现为
disabled,需评估)。

## Phase 1 — capture-size-list warmup(已落地,微调)

- `dlin_capture_sizes` 已读 prefill.bs + decode.bs 并逐大小预热。
- 待办(按 Phase 0 结果):
  - 若 prefill.bs 全量(42 个)启动太慢(~15-30min),提供合理默认子集 或 走 Phase 2 cache ship。
  - 若有非 MoE per-shape kernel 残留,改全路径 warmup 请求。
- 可选:把 warmup 挪到 **worker 侧**(model_runner,紧挨 cuda-graph capture,直接拿
  resolved `cuda_graph_config`),更贴 DLIN vLLM(worker-side warmup);当前 @warmup registry
  版在 tokenizer_manager 侧读 server_args,功能等价。

## Phase 2 — Cache shipping(中期,部署零启动 JIT)

warmup 一次,把 `~/.triton/cache`(+ DLEOL cache)打包随镜像分发:
- 构建脚本:离线跑全量 capture-size 预热 → 导出 cache tar。
- 部署解包 → 已知 capture 尺寸启动即零 JIT。
- 仍 per-shape:全新尺寸仍 JIT,但线上 capture 范围已全覆盖。

## Phase 3 — AOT 编译 / M-agnostic kernel(长期,根因解)

把 dleol/triton kernel 在 **build 时**预编译,运行时零 JIT:
- **路线 A(triton AOT)**:`triton.compile(kernel, signature)` → `.so`;验证 DLIN triton/dlcc 支持。
- **路线 B(M-agnostic kernel)**:fused MoE 把 M 作运行时参数(单次编译覆盖所有 M)→
  从根上消除 per-M JIT(kernel 侧深度工作)。
- **路线 C(vendor 进 `.so`)**:把 sglang DL kernel 全 AOT vendor(类已做的 vLLM kernel
  vendor / `_sgl_fa2_C`)。
- 参考:vLLM `_dl_C.so`(sdk-built,AOT 零 JIT)。

## 优先级

| 阶段 | 状态/工作量 | 收益 |
|---|---|---|
| Phase 0 盘点 | 1 次 profile | 确认 JIT 面(预计主要 fused MoE) |
| Phase 1 capture-list warmup | **已实现**;按 Phase 0 微调(全路径/子集) | capture 尺寸零首命中 |
| Phase 2 cache ship | 小(打包脚本) | 部署零启动 JIT |
| Phase 3 AOT / M-agnostic | 大(kernel/compiler) | 彻底零 JIT,任何形状 |

## 风险

- per-M JIT 是 dlcc/dleol 特性;Phase 1/2 是补偿,Phase 3 才是根因解。
- 全量 prefill.bs(42 尺寸)warmup 启动 ~15-30min(一次性,cache 后快)→ 需子集或 cache ship。
- 非 capture 尺寸的真实请求 M(prefill eager,不 round 到 capture 尺寸)仍可能 JIT ——
  这要看 sglang 是否对 prefill 做尺寸 bucketing;若没有,Phase 3(M-agnostic)才是彻底解。
