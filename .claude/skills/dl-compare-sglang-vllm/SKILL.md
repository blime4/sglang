---
name: dl-compare-sglang-vllm
description: Benchmark sglang vs vLLM (MRV1/MRV2 model runners, plain + MTP/DFlash spec decoding) on Denglin (DLIN) GPUs, same model + same GPUs, fair TPOT/speedup comparison. Use when comparing sglang and vLLM performance on DLIN, measuring spec-decoding speedup, or reproducing the DFlash/MTP investigation. Trigger keywords: dl-compare, sglang vs vllm, MRV1, MRV2, qwen3_next_mtp, DFlash bench, spec bench DLIN.
---

# dl-compare-sglang-vllm — sglang vs vLLM 对比基准 (DLIN)

**Goal:** 在同一台 DLIN 机器、同一模型、同一(组) GPU 上，公平对比 sglang 与 vLLM
（V1/V2 model runner、plain + MTP/DFlash spec decoding）的 TPOT / 加速比，并定位差异根因。

本 skill 固化了 2026-07 DFlash/MTP 调查中**全部验证过的配置和踩过的坑**。复用时
直接套用下文的命令模板，避免重复踩雷。

---

## 0. 关键概念速查

- **MRV1 / MRV2** = vLLM Model Runner V1 / V2。MRV2 由 `VLLM_USE_V2_MODEL_RUNNER=1`
  触发，走 FULL-CG + `split_graph` compile 路径。MRV1 是默认。
- **MTP** = 多 token 预测。vLLM 的 Qwen3.5 方法是 `qwen3_next_mtp`（**复用 target 自身
  当 draft**，不需要独立 MTP 头）。sglang 用 `FROZEN_KV_MTP`（frozen-KV context）。
- **DFlash** = block-diffusion draft。sglang 用 `speculative_algorithm='DFLASH'` +
  独立 draft 模型 `models-dl/Qwen3.5-35B-A3B-DFlash`。vLLM 的 `method='dflash'` 在
  DLIN 上**会崩**（见 §6）。
- **公平对比的金科玉律**：同一 prompt、同一 max_tokens、同一(组) GPU、**顺序跑**（每个
  配置一个 fresh 进程）、TPOT 取 best-of-N。spec 加速比 = plain_TPOT / spec_TPOT。
- **accept_len**（只对 spec 有意义）= `completion_tokens / spec_verify_ct`，是个计数，
  **不受 GPU 干扰污染**——是最可靠的 spec 质量指标。

---

## 1. 一次性环境准备

```bash
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
source sdk-dlop-07-13-20-30/env.sh          # 统一 SDK（NOT sdk-0401）
```

两个 python，**必须用绝对路径**（见 §6 坑①）：
- sglang: `.venv/bin/python`（Python 3.12）
- vLLM: **`.venv/bin/python`**（同一个 venv — `.venv` 已原生装好 vLLM 0.21.1.dev2 + DLIN triton 3.3.0 + `_dl_C.so` + dl plugins，**无需 overlay**）
  - ⚠️ **不要再用** `../venv-vllm021/bin/python` + `PYTHONPATH=vllm-new-overlay`：overlay 只遮蔽顶层 `vllm` 包、不覆盖 `vllm.v1.worker.*` 子模块 → 跑的是 0.21.0 未打补丁 core → 假性 gumbel/`_load_ptr`/penalty/"double dtype" 错误（这些都是 overlay 配置伪影，非真实 DLIN 缺陷）。见 [[dlin-vllm-correct-package-and-env-gotchas]]。
  - 运行 vLLM 必须作为**顶层脚本**（带 `if __name__ == '__main__'` guard，见坑⑪），且先 `source env.sh`（worker JIT 需要 `dlcc`）。

模型（已验证可用）：
- FP8: `/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/`（**sglang + vLLM 对比都用 FP8**，公平匹配精度）
- GPTQ-Int4: `/mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4/`（仅在 FP8 不可时备用）
- DFlash draft: `models-dl/Qwen3.5-35B-A3B-DFlash`

DLIN 通用 env（两个框架都建议开）：
```bash
export DLEOL_FLA_ENABLE_PINGPONG=1 DLEOL_FLA_UNROLL_COUNT=8 DLEOL_CACHE_SIZE=1024 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 2. GPU 管理（DLIN 独有，必读）

- **用 ≥16 的卡**（0-15 是同事的）。
- **DLIN 驱动泄漏**：被 kill 的进程**不释放显存**，卡卡在 12-26GB/100%。重置：
  `echo <sudo_pw> | sudo -S dlsmi -r -i <id>`（密码问用户；**不要写进脚本/git**）。
- **GPU 状态**：`dlsmi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
  （**NOT nvidia-smi**——DLIN 用 dlsmi）。
- **TP 需要 4 张卡**；vLLM 不带 `--tensor-parallel-size` 时默认 **TP1**（只用 1 张）。
  对比时**务必统一 TP**，否则绝对 TPOT 不可比（TP1 ≈ 2.5× 慢于 TP4）。
- **并行跑会互相干扰**（PCIe/CPU compile 竞争 + 残留显存）→ 顺序跑，每个配置 fresh 进程。

---

## 3. sglang 基准（plain / DFlash / MTP）

用 `.venv/bin/python`，离线 `sgl.Engine`（sglang 离线模式 spec 正常，不像 vLLM 会 hang）。

**通用 env**：
```bash
export SGLANG_DL_GDN_DLIN=1 SGLANG_DL_MOE_FUSED=1 SGLANG_DL_MOE_FUSED_MAX_M=32 \
       SGLANG_DL_FP8_Q2=1
```
（`SGLANG_DL_MOE_FUSED_MAX_M=32` 是关键：spec verify 的 M=block+1=9 要落在 fast-fused
MoE 路径上；默认 16 会漏掉 block=16 的 verify M=17。）

**Engine 关键参数**（plain / DFlash / MTP 三选一）：
```python
common = dict(model_path=MODEL, dtype="bfloat16", tp_size=4, attention_backend="fa3",
    page_size=16, context_length=4096, mem_fraction_static=0.60,
    disable_cuda_graph=False, disable_custom_all_reduce=True, trust_remote_code=True,
    cuda_graph_max_bs_decode=4, max_running_requests=4)
# DFlash（DLIN 上 block=8 最优，NOT 官方推荐的 16）
common.update(speculative_algorithm="DFLASH",
    speculative_draft_model_path="models-dl/Qwen3.5-35B-A3B-DFlash",
    speculative_dflash_block_size=8)
# 或 MTP：
common.update(speculative_algorithm="FROZEN_KV_MTP", speculative_num_steps=3,
    speculative_num_draft_tokens=4, speculative_eagle_topk=1)
```

**测 TPOT + accept**（warmup 3 次，best-of-3）：
```python
for _ in range(3): engine.generate(P, sampling_params={"max_new_tokens":4,"temperature":0})
best=9e9
for _ in range(3):
    t0=time.perf_counter()
    out=engine.generate(P, sampling_params={"max_new_tokens":N,"temperature":0})
    best=min(best, time.perf_counter()-t0)
meta = out["meta_info"]
accept = meta.get("completion_tokens",N) / meta.get("spec_verify_ct")  # spec only
print(f"TPOT={best/N*1000:.2f}ms accept={accept}")
```

**既有 harness**：`scripts/dl/benchrun_sglang.py`（vLLM `vllm bench run` 的 sglang 移植）、
`scripts/dl/dflash_correctness.py`、`scripts/dl/mtp_correctness.py`。

---

## 4. vLLM 基准（MRV1/MRV2 × plain/MTP）—— 必须用 serving 模式

**坑②（§6）**：vLLM **离线 `LLM()` 模式在 DLIN 上做 spec 会 hang/崩**（decode CG 捕获
device page fault）。**必须用 `vllm serve` + HTTP client**。

**坑③**：vLLM MRV1 在 DLIN 上撞 `ConstraintViolationError`（torch.compile 动态形状
guard）→ **DLIN 上只用 MRV2**（`VLLM_USE_V2_MODEL_RUNNER=1`）。

**坑④**：vLLM MTP 在 DLIN 上需 `--max-num-seqs 64`（默认 256 > 混合模型 206 个 Mamba
cache block → 崩）。

启动模板（TP1 MRV2 MTP，单卡）：
```bash
source sdk-dlop-07-13-20-30/env.sh   # worker JIT 需要 dlcc 在 PATH
CUDA_VISIBLE_DEVICES=20 VLLM_USE_V2_MODEL_RUNNER=1 \
  DLEOL_USE_CU_MQA_TILEKV=1 VLLM_MAX_MOE_CU_TOKENS=128 \
  .venv/bin/vllm serve \
  /mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/ \
  --port 8200 --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size 1 \
  --max-num-seqs 64 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,16],"max_cudagraph_capture_size":16}' \
  --trust-remote-code --served-model-name bench \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
```
- ⚠️ **用 `.venv/bin/vllm`（不要 `../venv-vllm021` + overlay）**——`.venv` 原生有完整 DLIN 栈，FP8+CG 直接跑通，无源码补丁。
- **必须 `source env.sh`**：vLLM worker 在首次生成时用 `dlcc` JIT 编译新 shape 的 kernel；不 source 会 `FileNotFoundError: 'dlcc'`。
- 离线 `LLM()` benchmark 必须是**带 `if __name__=='__main__'` guard 的顶层脚本**（见坑⑪），否则 `multiprocessing.spawn` 重导入 main → bootstrap RuntimeError。参考 `scripts/dl/vllm_features_only.py`。
- plain：去掉 `--speculative-config`。
- MRV1：去掉 `VLLM_USE_V2_MODEL_RUNNER=1`（DLIN 上大概率崩，见坑③）。
- TP4：`--tensor-parallel-size 4` + `CUDA_VISIBLE_DEVICES=20,21,22,23` + 4 张邻接卡。
- capture 很慢（~90s/batch）→ `max_cudagraph_capture_size=16` 限小，缩短启动到 ~3min。

**就绪判定**：轮询 `curl -s http://localhost:8200/health`（启动+捕获需 3-15min）。

**HTTP 测 TPOT**（OpenAI `/v1/completions`，warmup 3 + best-of-3）：
```python
import time, urllib.request, json
URL="http://localhost:8200/v1/completions"
def gen(mx):
    body=json.dumps({"model":"bench","prompt":PROMPT,"temperature":0,"max_tokens":mx}).encode()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(URL,data=body,
        headers={"Content-Type":"application/json"}),timeout=120).read())
    return r["usage"]["completion_tokens"]
for _ in range(3): gen(16)
best=9e9
for _ in range(3):
    t0=time.time(); n=gen(160); best=min(best,(time.time()-t0)/n*1000)
print(f"TPOT={best:.2f}ms")
```

---

## 5. 根因定位 / 分解诊断（spec 调到瓶颈时用）

**DFlash 每步 draft/verify 计时**（已提交，gated）：
`SGLANG_DL_DFLASH_TIMING=1` → 打印 `[dl-dflash-step] draft_ms=X verify_ms=Y cg=Z M=W`。
（默认关闭，不影响生产。）

**verify 内部分解**（clean，无 per-layer sync 污染）—— sglang qwen3_5.py 已内置差分模式：
```bash
# 三次跑，比较 verify_ms：normal / skip-attn / skip-moe
export SGLANG_DL_DFLASH_TIMING=1
# baseline:    verify_ms = V
# +SGLANG_DL_SKIP_ATTN=1: verify_ms = V_a  →  attention成本 = V - V_a
# +SGLANG_DL_SKIP_MOE =1: verify_ms = V_m  →  MoE成本     = V - V_m
```
（输出会错（零掉了 attn/MoE），但**计时有效**——这是干净的组件分解。）

**DLIN profiler**（kernel 级）：`dlpti_tools`（需完整 SDK env；非侵入式）。
**sglang trace**：`SGLANG_TORCH_PROFILER_DIR=/path`（scheduler 子进程抓 chrome trace）。

---

## 6. DLIN 对比踩坑清单（必读，省大量时间）

| # | 坑 | 规避 |
|---|---|---|
| ① | `source env.sh` 后 `python` 可能 = **Python 2.7** | 一律用 `.venv/bin/python` / `venv-vllm021/bin/python` 绝对路径；f-string 脚本在 py2 下静默 SyntaxError 像挂起 |
| ② | vLLM **离线 `LLM()` + spec** 在 DLIN 上 hang/崩（decode CG page fault）| 用 `vllm serve` + HTTP，不要离线 LLM() |
| ③ | vLLM **MRV1** 在 DLIN 撞 `ConstraintViolationError` | 用 MRV2（`VLLM_USE_V2_MODEL_RUNNER=1`） |
| ④ | vLLM MTP 默认 `max_num_seqs=256` > Mamba cache(206) → 崩 | `--max-num-seqs 64` |
| ⑤ | vLLM `method=dflash` 在 DLIN 崩（device page fault）| DLIN 上 spec 用 `qwen3_next_mtp`，不要 dflash |
| ⑥ | 杀掉的进程**不释放显存**（DLIN 驱动泄漏）| `echo <pw> \| sudo -S dlsmi -r -i <id>` |
| ⑦ | 并行多配置互相干扰（残留显存 + CPU compile 竞争）| 顺序跑，fresh 进程，独占邻接卡 |
| ⑧ | vLLM 默认 TP1 vs sglang TP4 → 绝对 TPOT 不可比 | 统一 TP；比**加速比**(spec/plain)更robust |
| ⑨ | sglang DFlash `block=16`（官方推荐）在 DLIN = 0.98×（无加速）| DLIN 用 **block=8** |
| ⑩ | 测量受 GPU 干扰时，绝对 TPOT 不可靠 | 看 **accept_len**（计数，不污染）+ 同窗口 plain/spec 比值 |

---

## 7. 结果解读（避免误判）

- **spec 加速比** = plain_TPOT / spec_TPOT。**务必同 TP、同 prompt、同卡、顺序跑**。
- **accept_len** 是 spec 质量金标准（计数，不受干扰）。DLIN 上 block=8：散文 ~4.2、
  代码 ~5.7-6.7、thinking-off ~5.65；官方 B200 block=8：5.4-5.9。
- **DLIN spec 天花板 ~1.2-1.3×**（已跨框架证实：sglang DFlash 1.28× ≈ vLLM MTP 1.17×）。
  根因 = verify forward 算力受限（45% GDN-extend + 42% MoE kernel），**不是适配 bug**。
  要破 1.3× 必须优化 DLIN verify 内核（见 `docs/dl/dflash-on-dlin-debug-blog.md` §5.14）。
- thinking 模式会**拉低 accept**（自由推理 draft 难预测）→ 关 thinking 或用结构化输出
  任务可提升 accept 与加速比。

---

## 8. 最小可复现：一页纸对比脚本骨架

```bash
# 0) reset + clean
for i in 20 21 22 23; do echo $SUDO_PW | sudo -S dlsmi -r -i $i; done
source sdk-dlop-07-13-20-30/env.sh
# 1) sglang plain TP4 + sglang DFlash block=8 TP4 (顺序, .venv/bin/python)
# 2) vLLM plain MRV2 + vLLM MTP MRV2 (vllm serve 或离线 LLM() 顶层脚本, TP1/TP4, **.venv/bin/python**, 无 overlay, 先 source env.sh)
# 3) 各自记录: TPOT(best-of-3), accept_len(spec only)
# 4) 汇总: spec/plain 加速比; 跨框架对比是否撞同一天花板
```

---

## 参考（本仓库内）

- `docs/dl/dflash-on-dlin-debug-blog.md` — 完整 DFlash/MTP DLIN 调查（§5 是本 skill 的事实依据）
- `docs/dl/sglang-vs-vllm-tp4-20260715-report.md` — sglang vs vLLM plain TP4 差距（host 开销）
- `scripts/dl/benchrun_sglang.py`, `scripts/dl/dflash_correctness.py`, `scripts/dl/mtp_correctness.py`
- `scripts/dl/compare_tp4.py` — sglang vs vLLM plain TP4 harness（spec 需自行扩展）
- 官方: [DFlash blog](https://z-lab.ai/projects/dflash/), [GitHub](https://github.com/z-lab/dflash)
