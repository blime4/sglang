---
title: "让 SGLang 跑在登临 GPU 上:运行 Qwen3.5-35B-A3B-FP8,并在 TP4 上对标 vLLM"
subtitle: "一篇关于把推理框架移植到非 NVIDIA GPU、并为 3 毫秒死磕一周的实战记录"
authors: "SGLang 登临(DLIN)适配团队"
date: 2026-07-17
tags: [sglang, dlin, 登临, 硬件适配, moe, fp8, 线性注意力, 性能]
model: Qwen3.5-35B-A3B-FP8
hardware: 登临 DLIN KS38(×4)
baseline: vLLM 0.21.1
---

# 让 SGLang 跑在登临 GPU 上
## 运行 Qwen3.5-35B-A3B-FP8,并在 TP4 上对标 vLLM

> *一篇关于把推理框架移植到非 NVIDIA GPU、并为 3 毫秒死磕一周的实战记录。*

**作者:**SGLang 登临(DLIN)适配团队　·　**日期:**2026 年 7 月　·　**模型:**Qwen3.5-35B-A3B-FP8　·　**硬件:**登临 DLIN KS38(4 卡)　·　**对标:**vLLM 0.21.1

---

## 摘要

我们将 SGLang 移植到登临(DLIN)GPU,使 **Qwen3.5-35B-A3B-FP8** ——一个混合线性注意力 + MoE、采用 blockwise FP8 量化的模型——达到正确、可对外提供服务的运行状态。工作分三个阶段推进:以 vLLM 为参考建立正确性、选择正确的 DLIN MoE kernel 路径以恢复吞吐、以及在 TP4 下对标 vLLM 的 decode 延迟。稳态下 SGLang 达到 **27.3 ms/token**,vLLM 为 **24.0 ms/token**;这 3.3 ms 的差距,我们用直接测量定位到了单一根因:两个引擎运行字节级相同的 GPU 原语,但 vLLM 的 `torch.compile` 应用了 Inductor 的 IR 级算子融合(`fuse_norm_quant`、`fuse_act_quant`),而 SGLang 在 DLIN 上暂时无法使用。我们已落地 compile 集成修复的第一阶段,并报告剩余路径。本文完整记录测量方法,并对过程中修正过的若干论断如实说明。

---

## 1. 背景

### 1.1 登临(DLIN)GPU

登临 GPU 是一类国产 GPGPU。它在**编程模型层面兼容 CUDA,但运行时与算子栈完全自研**:没有 cuBLAS、cuDNN、cuBLASLt,也没有预编译好的 FlashAttention 二进制。其软件环境由以下几部分构成:

- 自研 SDK,提供算子头文件(`<dldnn_ext.h>`、`<dlblasLt_ext.h>`);
- **DLEOL**,一个 JIT 编译虚拟机,在运行时把 Triton 和内部 IR 翻译成 GPU fatbin;
- 一套 profiling 工具链(`dlpti_tools`,对标 Nsight)和 `dlsmi`(对标 `nvidia-smi`)。

对一个推理框架而言,实际后果是:每一个性能关键算子——FP8 GEMM、分组 MoE、分页注意力、RMSNorm——都必须从 SDK 获取、经 DLEOL JIT 编译,或用登临编译器(`dlcc`)手写。高价值算子(分组 MoE、线性注意力)只有走 DLEOL-JIT 的 `_dl_C.so` op 才快,而这些 op 具有**上下文敏感的编译行为**:会根据调用上下文选择不同的 kernel 变体。

### 1.2 模型:Qwen3.5-35B-A3B-FP8

这不是一个 Llama 类的 dense 模型。它结合了三个要素,每一个都对应一整套需要适配的算子栈:

- **MoE 路由:**256 个 expert,每 token 激活 8 个,blockwise FP8 `[128,128]` 权重。每个 decode 步要做一次路由决策,加 8 次 per-expert FP8 GEMM。
- **混合注意力:**40 个 transformer 层里,**30 层是 GatedDeltaNet(GDN,线性/状态空间式注意力)**,只有 10 层是标准全注意力。每个 GDN 层除了因果 conv1d 和门控投影外,还要做一次循环状态更新(`dl_recurrent_gated_delta_rule`)。
- **FP8 量化:**权重 blockwise `[128,128]`;激活量化则是登临上大多数正确性"地雷"所在之处。

整体架构见**图 1**:每一个模块都必须映射到一个既存在、又足够快的 DLIN 算子。

![图 1 — Qwen3.5-35B-A3B-FP8 混合��构](figures/fig1-architecture.zh.svg)

*图 1. 40 层混合堆栈。两种注意力(10 层全注意力、30 层 GatedDeltaNet 线性注意力)都汇入一个共享的 FP8 MoE 模块。每个模块都需要对应的 DLIN 算子存在且足够快。*

因此,这不是一次"换个 attention backend"的工作,而是在新架构 + 新 GPU 上组装一套完整算子栈、并让它具备竞争力。

---

## 2. 算子映射

在谈性能之前,先谈"存在"。表 1 把每个架构原语映射到实际执行它的 DLIN 算子,以及 SGLang 获取它的途径。这部分集成工作占了整体工作的大头,却很少出现在适配文章里。

**表 1.** 算子集成映射。

| 原语 | DLIN 算子 | SGLang 获取方式 | 备注 |
|---|---|---|---|
| Decode 注意力 | `_vllm_fa2_C.so` / `paged_decode_attn` | `dlcc` 手写编译 | FlashAttention 源码 + 登临编译器 |
| GDN decode | `_dl_C.dl_recurrent_gated_delta_rule` | dlopen vLLM 的 `_dl_C.so` | DLEOL-JIT;约省 3 ms |
| GDN prefill | `_dl_C.dl_chunk_gated_delta_rule` | 同上 | `l2norm` 需在 op 外部做 |
| FP8 dense 线性 | `_dl_C.gptq_dlblas_gemmex` | 同上 | FP8 权重 + bf16 激活 |
| 融合 FP8 MoE | `_dl_C.invoke_fused_moe_opt` | 同上 | 分组 FP8 GEMM |
| Gemma RMSNorm | `_dl_C.gemma_rms_norm` | 同上 | 与 vLLM 共用 |
| 标准 RMSNorm | `sgl-kernel` dlcc kernel | 仓内自建 | — |
| TP all-reduce | NCCL(`pynccl`) | 框架自带 | 自定义 all-reduce 在 DLIN 上 codegen 失败 |

一个关键细节:SGLang **并不编译** `_dl_C.so`,而是直接 `dlopen` vLLM 虚拟环境里那一份共享库(`fp8_utils.py::_ensure_dl_C()`)。`nm` 可证两引擎调用的是相同符号。这是第 5 节核心论断的基础——**SGLang 与 vLLM 执行字节级相同的 GPU 原语**——也正是剩余延迟差距必然落在"原语够不到的地方"的原因。

---

## 3. 第一阶段 —— 正确性

首次启动产出了确定性的乱码:*"the a capital of the capital of…"*,而 vLLM 返回 *"Paris."*。"稳定地错"比"崩溃"更难——它说明计算在跑,只是算错了。我们定位到四个相互独立的缺陷,每一个都附带一条可迁移的经验。

### 3.1 写死的旋转位置编码风格(所有质量问题的总根因)

SGLang 的 Qwen3 代码写死了旋转变体:

```python
# qwen3_5.py — 修改前
is_neox_style = True
```

这对 Qwen2/Qwen3 **纯文本**模型是对的(NeoX 旋转把 head 维度对半劈)。但这个模型尽管名为 Qwen3.5,实际是 **Qwen3-VL 多模态架构**,config 里有 `rope_parameters.mrope_interleaved: True`,需要的是**交错(interleaved)**旋转——相邻维度两两一组,而不是对半劈。用 NeoX 会把旋转作用到错误的维度对上,注意力全错,输出退化成复读。已落地的修复:

```python
# qwen3_5.py:808 — 修改后
# DL: model has mrope_interleaved=True (Qwen3-VL); needs interleaved rotary
# (is_neox_style=False), not NeoX (True).
is_neox_style = (
    not getattr(config, "rope_parameters", {}).get("mrope_interleaved", False)
)
```

这一个布尔量是所有配置下全部质量症状的共同根因。*经验:*模型名字会误导架构判断;旋转风格标志藏在 `config.json` 的 `rope_parameters` 嵌套结构里。

### 3.2 一个"防御性"的 `.contiguous()` 引发段错误

FP8 GEMM 路径(`quant_type=2`)直接段错误退出(exit −11)。SGLang 在权重上加了防御性的 `.contiguous()`;vLLM 则直接传**非连续**的 `weight.t()` 转置视图。登临 kernel 硬编码了转置视图的 stride;一旦把张量拍平,stride 就对不上,kernel 越界访问。去掉这条路径上所有的 `.contiguous()`,崩溃消失,**还顺带省了约 4 ms TPOT**。一个"看起来更稳妥"的调用,恰恰是缺陷所在。

### 3.3 自定义 all-reduce 在张量并行下崩溃

开启张量并行后,在 `cross_device_reduce_1stage<bfloat16,2>` 崩溃,报 `HC_CUK Error=28`:登临 codegen 无法生成这个自定义 all-reduce kernel。vLLM 在登临上跑 TP 用的是 NCCL(`use_custom_allreduce=False`);SGLang 采用同样解法:

```
--disable-custom-all-reduce    # 走 NCCL,与 vLLM 一致
```

### 3.4 FP8 融合 MoE:输出乱码(最深的那个)

融合 MoE 路径(`invoke_fused_moe_opt`)输出乱码;bf16-bmm 回退路径(反量化成 bf16 再 `torch.bmm`)返回 *"Paris"*——但代价是慢 3 倍。我们用一个决定性的技术定位根因:**与参考实现做字节级 tensor 对比**。我们从 SGLang 进程内部拉起 vLLM(对 DL op 垫片做插桩),把两引擎喂给 `invoke_fused_moe_opt` 的张量都 dump 出来逐字节比较。

**表 2.** 喂给 `invoke_fused_moe_opt` 的输入,SGLang vs. vLLM。

| 张量 | SGLang | vLLM |
|---|---|---|
| `w13_weight` | (256,256,2048) fp8,stride (524288,2048,1) | **完全一致** |
| `w13_scale` | (256,4,16) fp32,stride (64,16,1) | **完全一致** |
| 激活 `x` | (M,2048) bf16 | **完全一致** |
| **输出** | 乱码 | "Paris" |

同一个 op、同一份 `.so`、相同字节输入——输出却不同。缺陷不可能在 op 里,只能在调用前的准备步骤。差异追溯到 MoE 对齐:vLLM 的 DL 融合 MoE 有两种路由模式——大 M 用 `moe_align_block_size(ignore_invalid_experts=True)`,小 M(decode)用 `use_moe_cu`(平凡派发张量,路由在 op 内部完成)。SGLang 永远走自己的 Triton `moe_align` 且**不设** `ignore_invalid_experts`,产生了垃圾 expert id(最大值 1.033×10⁹)。op 于是从无效 expert 索引去 gather。*经验:*对"算错了"类缺陷,在臆测 op 之前,先证明输入与参考实现逐字节一致。

修完这四个,SGLang 的输出与 vLLM 逐 token 对齐。

---

## 4. 第二阶段 —— 吞吐

正确输出到来时,TPOT 约 76 ms/token。优化是一次沿着 MoE 路径树往下选最快分支的走法:

```
GEMMEX=2  (gptq_dlblas_gemmex,CG 安全,无激活量化,单独 gather)  ≈76 ms  基线
   │  切到分组 GEMM 快路径
   ▼
融合 MoE  use_moe_cu  (invoke_fused_moe_opt)                  ≈28 ms
```

`dlPTI` profiler 解释了 GEMMEX 的代价。GEMMEX 下 decode 的 kernel 占比里,单独的 `vectorized_gather_kernel` 占了 **22.6%(约 6 ms)**——即单独的 `weight[expert_id]` gather。融合路径把这个 gather 并进了 GEMM(变成 `take_b`,2.5%);MoE 时间从约 14 ms 降到约 7.8 ms,基本追平 vLLM 的 8.64 ms。收益不是"GEMM 更快",而是"少做一次 6 ms 的 gather"。

融合路径用 env 门控,默认路径不受影响:

```python
# fp8.py — DLIN 融合 blockwise FP8 MoE,env 门控
_DL_MOE_FUSED_MAX_M = int(os.environ.get("SGLANG_DL_MOE_FUSED_MAX_M", "16"))
if (
    is_dlin()
    and os.environ.get("SGLANG_DL_MOE_FUSED", "0") == "1"
    and x.shape[0] <= _DL_MOE_FUSED_MAX_M     # decode M=1, verify M≈9, 短 prefill
):
    ...  # invoke_fused_moe_opt 快路径
```

`FUSED_MAX_M` 上限定在 16 而非更高,是因为一个真实的登临编译器缺陷:`invoke_fused_moe_opt` 在 prefill `M ≥ ~100` 时触发 DLEOL assert(`tu_program.cc:625` stride 对齐 → SIGSEGV)。稳定的 serving 配方把融合路径与**分块 prefill**(`chunked_prefill_size=16`)配合,让每次 forward 的 MoE batch 保持 `M ≤ 16`,远离该 assert。长 prefill 从 33 s 降到暖态 5 s,输出连贯。

---

## 5. 第三阶段 —— TP4 延迟对标

第二阶段后,SGLang 到了约 27 ms/token;vLLM 是 24 ms。**定位这剩余的 3 ms 耗掉了本周剩下的全部时间,而且要求我们推翻自己的第一个假设。**

### 5.1 头条测量

用对比脚本(`scripts/dl/compare_tp4.py`)在两引擎上跑完全相同的 workload——prompt 缩成 "Hi"、生成 512 token,把 TTFT 稀释到约 0.4 ms/token——纯稳态 decode 给出:

**表 3.** TP4 decode-only TPOT。

| 引擎 | TPOT | 吞吐 |
|---|---|---|
| SGLang(TP4,CUDA Graph,NCCL) | **27.3 ms** | 36.6 tok/s |
| vLLM(TP4,CUDA Graph,NCCL) | **24.0 ms** | 41.7 tok/s |
| 差距 | **+3.3 ms** | 1.14× |

### 5.2 假设一 —— "差距在 host 侧"(错误)

我们拼出了一个看似无懈可击的"GPU 已持平"论证:

1. **相同原语:**`nm -D _dl_C.so` 显示两引擎调用相同符号。
2. **相同 forward 时间:**CUDA event 包住 `graph.replay()`,在 *scheduler 进程内、正确的 stream* 上测,两边都是 **20.7 ms**。
3. **差分校验:**`SGLANG_DL_SKIP_MOE=1` 跳过 MoE → MoE = 7.8 ms,非 MoE = 12.9 ms,加起来正好 20.7 ms,没有隐藏的 GPU 浪费。

逻辑闭环:既然 GPU 相同,3.3 ms 只能在 host 侧。我们画出一步 decode 的完整 host 栈,识别出六个开销点,最大的是 `load_batch`/`fill_from`——它把 batch 拷进静态 CUDA Graph 输入缓冲(每步 6–7 次 `copy_`)。随后我们花了好几天给六个点逐一打补丁:bs=1 的 `load_batch` fast-path、双缓冲原地 `seq_lens`、异步 `.tolist()`、async-replay worker 线程、multi-step decode。**没有一个能把 TPOT 动哪怕 0.5 ms。** multi-step 单次到过 24.95 ms,但稳健复测 6 次平均 30 ms;那个 24.95 是噪声。

### 5.3 反转

当所有合理的修复都失效时,该质疑的不是修复,而是**测量地基**。直接测 host:

**表 4.** 每步 host 开销。

| 组件 | SGLang | vLLM |
|---|---|---|
| `pop_and_process`(中位数) | 0.7 ms | ~0.7 ms |
| `run_batch`(非 replay 部分) | ~1.5 ms | ~1.5 ms |
| recv / get_batch | ~2.8 ms | ~2.8 ms |
| **host 合计** | **~6 ms** | **~6 ms** |

**SGLang 的 host 开销与 vLLM 持平——甚至略低。**"host 侧"假设是错的。3.3 ms 必然在 GPU forward,可我们明明测出两边 forward 都是 20.7 ms。

### 5.4 盲区 —— `torch.compile` 的 IR 级融合

矛盾由 vLLM 的启动日志解开:

```
Enabled custom fusions: norm_quant, act_quant, rope_kvcache_cat_mla
compilation_config={'backend': 'inductor',
  'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True}}
torch.compile and initial profiling/warmup run together took 15.55s
```

vLLM 跑了 `torch.compile`(Inductor),做**IR 级算子融合**:

- `fuse_norm_quant` 把 RMSNorm + FP8 量化合并成一个 kernel(少一次访存、少一次 launch);
- `fuse_act_quant` 把激活 + 量化合并成一个 kernel。

这些融合 kernel **不在 `_dl_C.so` 里**——`nm` 可证 `.so` 里只有一个独立的 `gemma_rms_norm`(SGLang 也在用)。融合是 Inductor 在编译期*生成*出来的。我们那个 20.7 ms 对"共享原语"是对的,但 vLLM 的编译后图并没有按这个数量去跑那些原语——它跑的是*更少的、融合后的* kernel,effective forward 是 **约 18 ms**,不是 20.7。修正后的闭环:

```
SGLang  TPOT = GPU_forward(20.7, 无融合) + host(6)   = 27 ms
vLLM    TPOT = GPU_forward(~18,  有融合) + host(6)   = 24 ms
                                       ▲
                          全部 2–3 ms 差距都在这里
```

**图 2** 把归因讲精确:host 与 MoE 在两引擎间持平;差距集中在 GPU 非 MoE 段,也就是融合所作用的 norm/quant 算子所在之处。

![图 2 — TP4 decode 延迟分解](figures/fig2-latency-decomposition.zh.svg)

*图 2. 单步 decode 拆解。host 开销与 MoE kernel 在两引擎间持平(同一份 `_dl_C.so`);全部 3.3 ms 差距落在 GPU 非 MoE 段——即 vLLM 通过 `torch.compile`(`norm_quant` / `act_quant`)融合、而 SGLang 暂时拿不到的 RMSNorm + FP8 量化算子。*

### 5.5 为什么 SGLang 拿不到这些融合

因为 SGLang 的 `torch.compile` **在登临上一开就崩**:

```
enable_torch_compile=True
  → triton_kernel_wrap ValueError
  → GDN conv kernel 期望 USE_GDC;SGLang 的调用没传
```

vLLM 之所以能在登临上成功编译,是因为它带了一整套 **`dl_platform_plugin`**,专门处理登临特有的编译问题(kernel 签名、PDL primitive、算子注册)。SGLang 没有等价物。所以差距不是"登临不支持 compile"——vLLM 已经证伪了这点——而是**"SGLang 缺少登临的 compile 集成层"**。

### 5.6 测量方法

为可复现,每个数字按如下方式产出,包括两个陷阱。

- **Decode-only TPOT:**prompt "Hi",生成 512 token,TTFT 稀释到约 0.4 ms/token。
- **纯 GPU forward:**CUDA event 包住 `graph.replay()`,在 **scheduler 子进程内**测(`SGLANG_DL_TIME_REPLAY=1`),绝不在主进程测。*陷阱:*多进程框架里主进程的 CUDA event 不可靠——GPU 工作发生在子进程。从错误的进程测,就是"证明"了并不存在的 GPU 持平。
- **组件分解:**`SGLANG_DL_SKIP_MOE` / `SKIP_ATTN` / `SKIP_SHARED` 差分(基线 − 跳过 = 组件开销)。
- **vLLM 参考:**`VLLM_DL_TIME_REPLAY=1` + `VLLM_TORCH_PROFILER_DIR`,并在启动日志上 `grep 'compil|inductor|fuse'` 确认融合已启用。

---

## 6. 优化日志与 JIT bug 分类

在 host 侧那段弯路里,我们跑了 24 个优化实验。下面摘要它们,不是因为它们成功了——大多没成功——而是因为**失败模式本身定位了真正的阻塞点**。

**表 5.** 优化实验(节选)。

| # | 实验 | 结果 | 失败模式 |
|---|---|---|---|
| 1–3 | MoE block size(M=16/48/64)、page size | 无变化 / 更差 | 选了相同 JIT 变体 |
| 6–7 | shared-expert 融合;per-expert GEMMEX | 更慢 | expert/循环开销 |
| 8–9 | 融合 `silu_and_mul`(Triton / vLLM `_C`) | **崩溃** | DLEOL JIT |
| 10–13 | `use_moe_cu`(+ relaxed / tc_piecewise CG) | **崩溃** | DLEOL JIT |
| 14–16 | 移植 vLLM `fused_experts_impl` | 更慢 / **崩溃** | DLEOL JIT |
| 22–23 | bf16 beta 输出;CG 下 `_C.silu_and_mul` | **崩溃** | DLEOL JIT |
| 24 | `_C.silu_and_mul` 独立微基准 | **OK** ✅ | op 本身没问题 |

**横贯性发现:**op 在隔离下正常、在模型里崩溃——每个融合变体在独立微基准(随机权重、任意 topk 模式、任意 scale)下都通过(max error 0),可一旦进入 `torch.distributed` 初始化的模型上下文就崩。三类 DLEOL-JIT 崩溃签名反复出现:

**表 6.** DLEOL JIT 失败分类。

| 类别 | 症状 | 触发 |
|---|---|---|
| `K%VEC_K` static assert | 编译期 assert | 改变 JIT key(block size、`mul_routed_weight`、fused-experts 配置) |
| Device page fault | 运行时越界访问 | 新 Triton 变体 / `use_moe_cu` / dtype 变化 |
| `cudaErrorInvalidAddressSpace` | capture 下不支持的内存操作 | `use_moe_cu` + CUDA-graph capture |

决定性的阻塞点是 `per_token_group_quant_8bit_v2.cuh:396`,它把 `kUsePDL = true` 写死,没查 `is_arch_support_pdl()`。PDL(Programmatic Dependent Launch)kernel 无法被 SGLang 的原生 CUDA graph 捕获,于是快的 `use_moe_cu` 路径被拒。vLLM 跑*同一个二进制*,但走 `torch.compile` 驱动的 capture,被接受。这正是为什么修复方向是 compile 集成、而非继续调 kernel:compile 驱动的 capture,正是 vLLM 同时拿到 `use_moe_cu` 与 norm/quant 融合的路径。

---

## 7. 如实更正

一份可信的报告要更正自己的论断。早先的几个说法,事后证明是不公平测量或损坏输出的假象。当前、已验证的状态如下:

**表 7.** 论断审计。

| 早先的论断 | 裁定 | 实际真相 |
|---|---|---|
| "SGLang decode 是 vLLM 的 1.45 倍" | 推翻 | 那是 SGLang 开 CG 对 vLLM **eager**。双方都开 CG:**vLLM decode 快约 1.4–1.5 倍。** |
| "NGRAM 投机解码达 35–40 tok/s,是 vLLM 的 3 倍" | 推翻 | 吞吐是真的,但测在**垃圾输出**上(spec-verify 复述了 prompt,抬高 n-gram 命中率)。不是有效加速。 |
| "SGLang 全面超过 vLLM" | 推翻 | SGLang 仅在两个窄口径占优:低 batch 下 SGLang-CG 对 vLLM-eager、以及 **batch 8 聚合吞吐(≈ 持平)**。decode 延迟 vLLM 领先。 |
| "CUDA Graph 是净负收益,用 eager" | 本模型上错误 | 35B 混合模型上 CG **净正收益**(B=1 eager 7.2 → CG 16.7 tok/s)。早先结论来自 1.7B dense 模型。 |
| "GDN 循环 kernel 是瓶颈" | 推翻 | `dlPTI` 显示 GDN 循环约 0.5 ms(1.3%)。MoE 修复后,剩余差距是 compile 融合。 |

**净立场:**SGLang 在登临上能正确运行 Qwen3.5-35B-A3B-FP8、能对外提供服务,TP4 decode 延迟在 vLLM 的约 14% 以内,聚合吞吐持平。我们*尚未*在延迟上超过 vLLM,但精确知道原因:`torch.compile` 融合差距。投机解码的吞吐数字在 verify 路径回归修好之前,都带明确的质量星号。

---

## 8. 修复路径 —— DLIN `torch.compile` 集成

既然根因是"SGLang 的 compile 在登临上崩",唯一能真正闭合差距的路径,就是让 compile 跑通,从而继承 `norm_quant`/`act_quant` 融合。预期收益:GPU forward 20.7 → 约 18 ms,叠加 SGLang 略低的 host,TPOT 进 23–24 ms 区间——与 vLLM 竞争。**图 3** 把这项工作放在优化轨迹上定位。

![图 3 — GPU forward 优化轨迹](figures/fig3-optimization-journey.zh.svg)

*图 3. GPU forward 时间沿优化轨迹的变化。fused MoE 修复(今天落地,20.7 ms)去掉了 GEMMEX 的单独 weight-gather;20.7 → 约 18 ms 这一步,是 `torch.compile` 提供给 vLLM、而 DLIN compile 集成层将为 SGLang 解锁的 norm/quant 融合。*

### 8.1 第一阶段(已落地)—— compile 不再崩

解决了四个阻塞点:

1. **GDN conv 的 `USE_GDC` 签名。**Triton kernel 用了 `**pdl_kwargs`(动态 dict);Inductor 的严格签名检查拒绝它。在 GDN、Mamba、elementwise、FP8 kernel 上全部改成显式 `USE_GDC=...` constexpr。
2. **`gdc_wait` / `gdc_launch_dependents` 桩。**登临 Triton 没有这两个 PDL primitive;Inductor 解析 kernel AST 时 `getattr` 它们并崩。加了 `@triton.jit` 的 no-op device-function 桩。
3. **DLIN 自定义 op 的 FakeTensor(抽象)实现。**没有它们,Inductor 会*分解* DLIN op(图膨胀:capture 306 s,decode 82 ms 对比基线 27 ms)。有了它们,Inductor 把调好的 DLIN kernel 当 opaque 保留:

   ```python
   # dl_compile_meta.py — 把 DLIN _dl_C op 注册为对 Inductor opaque
   def dl_register_meta():
       _ensure_dl_C()
       from torch.library import register_fake
       register_fake("_dl_C::gptq_dlblas_gemmex")(
           lambda input, weight, scale_inv, scale_inv2, qt, bit:
               input.new_empty(input.shape[0], weight.shape[1]))
       register_fake("_dl_C::gemma_rms_norm")(...)            # 原地
       register_fake("_dl_C::invoke_fused_moe_opt")(...)       # 分组 FP8 MoE
       register_fake("_dl_C::dl_recurrent_gated_delta_rule")(...)  # 线性注意力
       ...
   ```

   **结果:**`enable_torch_compile=True` 现在能编译、能抓 CUDA Graph、能跑出正确输出。

### 8.2 第二阶段(进行中)—— 让编译后的图变快

第一阶段止步于"不崩";编译后的图仍是约 81 ms(约等于 eager)。三个已定位原因:

- **compile ↔ CUDA Graph 没接好:**CG 抓到的还是 eager 路径,不是编译后的 lean 图。
- **缺融合 pass:**SGLang 的编译框架没有 vLLM 的 `norm_quant`/`act_quant` Inductor pass(它们长在 `dl_platform_plugin` 里)。需要移植。
- **残余 graph break:**部分 `_dl_C` op 仍缺 fake 实现。

### 8.3 备选方案

每次 compile 迭代都要一次模型加载(共享存储争用下 10–30 分钟)。若太慢,可走**路线 B**:请登临 kernel 团队把融合 kernel(norm+FP8 量化、act+量化)做成 `_dl_C.so` 里的独立 op,SGLang 直接调用,绕开 `torch.compile`。

---

## 9. 已验证的 serving 配方

下面的配置是登临 KS38 / TP4 上已验证、稳定的 serving 设置(长时间在线运行无崩溃):

```bash
DLEOL_CACHE_SIZE=1024 \
SGLANG_DL_MOE_FUSED=1 \
SGLANG_DL_MOE_FUSED_MAX_M=16 \
SGLANG_DL_FP8_Q2=1 \
SGLANG_DL_GDN_DLIN=1 \
DLEOL_FLA_ENABLE_PINGPONG=1 \
DLEOL_FLA_UNROLL_COUNT=8 \
python -m sglang.launch_server \
  --model-path /mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ \
  --tp 4 \
  --attention-backend fa3 \
  --page-size 16 \
  --chunked-prefill-size 16 \        # 让 MoE M≤16 → 避开 DLEOL assert
  --disable-custom-all-reduce \      # 走 NCCL,与 vLLM 一致
  --mem-fraction-static 0.60
  # CUDA Graph 默认开启(本混合模型上净正收益)
```

复现与基准脚本:`scripts/dl/compare_tp4.py`(头对头)、`scripts/dl/ttft_tpot.py`(延迟)。诊断环境开关(均默认关):`SGLANG_DL_TIME_REPLAY`、`SGLANG_DL_SKIP_MOE/ATTN/SHARED`、`SGLANG_DL_LAYER_TIMING`。

---

## 10. 工程经验

1. **一套自洽的论证加上 profile 数据,仍可能是错的。**"两引擎用同一份 `.so`,所以 forward 时间相等"——逻辑成立,但建立在一个未经验证的前提上(两引擎实际跑同一批 kernel)。当所有合理修复都失效时,先质疑测量地基。
2. **对"算错了"类缺陷,先和参考实现做输入的逐字节对比。**tensor-dump 工具一下午就解决了最难的正确性 bug,之前却是几小时的错误猜测。
3. **防御性代码常是移植性缺陷。**`.contiguous()` 之类"更稳妥"的调用,经常破坏那些硬编码 stride/布局假设的 kernel。
4. **在新 GPU 上,差距往往在 compile 集成层,而非 kernel。**登临的原语是够的——vLLM 已证明。SGLang 的欠缺,是教会 Inductor 这些登临 op 长什么样。
5. **对不公平基准测试保持警惕。**本项目里每一个后来被推翻的"我们超过 vLLM"数字,都是不对等比较(CG 对 eager,或在损坏输出上的吞吐)。若某个数字好得过分,回头查分母。

---

## 11. 路���图

- **`torch.compile` 第二阶段:**接通 compile↔CG、移植 `norm_quant`/`act_quant` 融合 pass、补齐残余 graph break——目标 GPU forward 约 18 ms,TPOT 23–24 ms。
- **投机解码质量:**NGRAM/MTP 的 verify 路径在本混合架构上仍有 prompt-复述回归,这会卡住任何投机解码的吞吐论断。
- **上游化:**env 门控的 DLIN 派发、算子映射、compile 集成脚手架都带标记(`# DL begin/end`),保持可 grep、能在上游同步中存活。

---

## 附录 A —— 产出物

- **代码:**分支 `dl-main`,五个 commit(正确性 / FP8-MoE 性能 / host 开销 / 一键 TP4 serve / compile 第一阶段)。
- **报告:**`docs/dl/sglang-vs-vllm-tp4-status-20260716.md`(终版);`docs/dl/sglang-vllm-tp4-gap-report.md`(组件级 GPU 分解与完整 24 实验日志);`docs/dl/qwen3.5-dlin-code-stack.md`(算子映射)。
- **配图:**`docs/dl/figures/fig1-architecture.zh.svg`、`fig2-latency-decomposition.zh.svg`、`fig3-optimization-journey.zh.svg`(英文版同名去 `.zh`)。
- **脚本:**`scripts/dl/compare_tp4.py`、`scripts/dl/ttft_tpot.py`。

---

*小结。SGLang 在登临 GPU 上能正确运行 Qwen3.5-35B-A3B-FP8 并对外提供服务;GPU 原语与 vLLM 完全相同;剩余的 decode 延迟差距,是 SGLang 的 DLIN compile 集成层尚未提供的 `torch.compile` IR 级融合。闭合它是一项工程任务,不是硬件限制——而且正在进行中。*
