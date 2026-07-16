# sglang vs vLLM TP4 性能差距 —— 超详细深度版（代码栈 + 大白话 + 30 个优化自问自答）

> 配套文档：[`sglang-vs-vllm-tp4-20260715-report.md`](./sglang-vs-vllm-tp4-20260715-report.md)（结论版）
> 本篇在结论版基础上，把"差距在 host 侧"这件事**拆到代码行级别**，用简明代码栈 + 大白话讲清楚每一步开销从哪来，
> 然后深度思考 **30 个可落地的优化点**，每个都做"自问自答"（问现状 / 答做法 / 估收益 / 标风险）。
>
> 日期：2026-07-16 ｜ 模型：Qwen3.5-35B-A3B-FP8 / TP4 / DLIN KS38 ｜ 分支：`dl-main`

---

## 0. 一句话结论（先记牢）

> GPU 内核 sglang 和 vLLM 完全一样（同一份 `_dl_C.so`，forward 都 ≈20.7ms）。
> sglang TP4 decode TPOT = **27.32ms**，vLLM = **23.98ms**，差 **3.34ms**，**100% 来自 sglang 每步的 host 侧开销**。
> 也就是说：**打平 vLLM 的钥匙不在 GPU，而在 CPU 那一侧的 Python / 拷贝 / IPC / 同步**。

大白话：vLLM 让 GPU 一刻不停（host gap ≈ 0ms），sglang 每生成一个 token 就被 CPU 拽住 ~6ms（vLLM 只要 ~3ms），还偶发 27ms 大卡顿。我们要做的，就是把这些"拽住 GPU 的 CPU 手"一只一只松开。

---

## 1. 为什么说"GPU 已经打平"（把地基夯实，省得后面被带偏）

历史上有过多次反复（perf-gap 文档 §7.21→§7.24→§7.27→§7.30→§7.57），这里给最终定论：

| 维度 | sglang | vLLM | 是否还有 GPU 优化空间 |
|---|---|---|---|
| Decode forward（CUDA graph replay） | **20.7ms** | ≈20.7ms | ❌ 已无（同 `_dl_C.so`） |
| MoE kernel | `invoke_fused_moe_opt` | `invoke_fused_moe_opt`（同一个） | ❌ |
| GDN kernel | `dl_recurrent_gated_delta_rule` | 同 | ❌ |
| block size (BM/BN/BK) | 全扫 16~128 | — | ❌ 0ms 差异 |

- `nm -D _dl_C.so` 实证：两个引擎共用 `invoke_fused_moe_opt` / `dl_recurrent_gated_delta_rule` / `gptq_dlblas_gemmex`。
- `SGLANG_DL_SKIP_MOE=1` 差分法：MoE = 7.8ms，非 MoE = 12.9ms，加起来正好 20.7ms，没有"隐藏的 GPU 浪费"。
- 旧的 §7.57"1.8ms gap"是用旧 `.so` 测的（版本差异），换了 Jul-14 新 `.so` 后 GPU 已对齐到 20.7ms 双方一致。

**所以：本文所有 30 个优化点都瞄准 host 侧。GPU 侧除非 DLIN 出新 kernel，否则不动。**

---

## 2. sglang 一步 decode 到底在 CPU 上干了啥（全栈地图）

先把"一步"的代码栈画出来。overlap scheduler 开启时的真实路径（这是 serving 默认）：

```
Scheduler.event_loop_overlap()            # scheduler.py:1514  每步循环
 ├─ recv_requests / process_input_requests
 ├─ get_next_batch_to_run()
 │    └─ ScheduleBatch.prepare_for_decode()        # schedule_batch.py:2602  ← 开销点①
 │         ├─ alloc_for_decode(token_per_req=1)    # KV slot 分配
 │         └─ seq_lens = seq_lens + 1              # :2690  overlap 下新建 3 个张量 ①
 ├─ run_batch(batch)                               # scheduler.py:3126
 │    ├─ resolve_forward_inputs(batch, future_map) # overlap_utils.py:65  取上一步 token
 │    ├─ model_worker.forward_batch_generation
 │    │    └─ ModelRunner.forward → decode_cuda_graph_runner.execute
 │    │         └─ load_batch()                    # decode_cuda_graph_runner.py:866  ← 开销点②(#1元凶)
 │    │              └─ buffer_registry.fill_from  # cuda_graph_buffer_registry.py:379
 │    │                   每步把 6~7 个 slot copy_ 进静态图输入缓冲
 │    │         └─ graph.replay()                  # 20.7ms，GPU 工作，最重但不归我们管
 │    │         └─ sample() argmax(151936)         # sampler.py:178（在图外）
 │    ├─ future_map.publish(seq_lens+1)            # overlap_utils.py:314  dtype cast + scatter ③
 │    ├─ future_map.stash(next_token_ids)          # overlap_utils.py:325  int32→int64 cast + scatter ③
 │    └─ (DL) 跳过 copy_to_cpu（非 logprob）       # scheduler.py:3216
 ├─ result_queue.append((batch.copy(), result))    # scheduler.py:1552  浅拷贝快照 ④
 ├─ pop_and_process() → process_batch_result(prev) # 处理"上一步"的结果
 │    └─ batch_result_processor.process_batch_result_decode()  # batch_result_processor.py:636  ← 开销点⑤
 │         ├─ next_token_ids.tolist()              # :780  ★ 每步唯一的强制 GPU↔CPU sync
 │         ├─ for req: update_finish_state / time_stats ...   per-req Python 循环
 │         └─ output_streamer.stream_output()      # output_streamer.py:117  ← 开销点⑥(IPC)
 │              └─ to_payload() ~30 字段 dataclass  # :517
 │              └─ send_to_detokenizer (pickle+ZMQ) # :163  跨进程
 └─ launch_batch_sample_if_needed()
```

**对照 vLLM**：vLLM 是单进程、tight loop，sampling/result/IPC 都在同一个进程里、且与 GPU 重叠得几乎完美（event 测 host gap ≈ 0ms）。sglang 多了一层"多进程 + Python event loop 驱动 + 每步重新填图缓冲 + 每步 tolist 同步 + 每步 IPC pickle"，这些加起来就是那 3.34ms。

---

## 3. 六大开销点逐一精讲（代码栈 + 大白话）

### 开销点 ① `prepare_for_decode`：overlap 下每步新建 3 个张量

代码栈 `schedule_batch.py:2690-2697`：
```python
if self.enable_overlap:
    self.seq_lens      = self.seq_lens + 1        # 新张量（拷贝）
    self.seq_lens_cpu  = self.seq_lens_cpu + 1    # 新张量（CPU 拷贝）
    self.orig_seq_lens = self.orig_seq_lens + 1   # 新张量
else:
    self.seq_lens.add_(1)                          # 原地，零拷贝
```

大白话：非 overlap 模式用 `add_`（原地加，不分配新内存）；overlap 模式怕"正在算的上一步"读到被改坏的 `seq_lens`，于是 `+1` 造一个**新**张量。一次造 3 个，32 步就 96 次 `aten::add`(出新张量)。这是为了正确性的"防御性拷贝"。

### 开销点 ② `load_batch` / `fill_from`：每步把 batch 塞进静态图缓冲 —— **#1 元凶**

代码栈 `decode_cuda_graph_runner.py:866 / 876-877 / 912`：
```python
def load_batch(self, forward_batch, ...):
    self.buffers.input_ids[:n].copy_(forward_batch.input_ids)   # :876
    self.buffers.positions[:n].copy_(forward_batch.positions)   # :877
    ...
    self.buffer_registry.fill_from(forward_batch, raw_bs=raw_bs) # :912
```
`cuda_graph_buffer_registry.py:379 fill_from` 按 dtype 分桶后 `torch._foreach_copy_`，被拷的 slot（`:559-601`）有：`input_ids, positions, out_cache_loc, req_pool_indices, seq_lens, seq_lens_cpu, mrope_positions`。

大白话：CUDA graph 是"录好的磁带"，只能读固定的输入缓冲地址。所以每步 replay 前，必须把"这一步真正要算的数据"拷进那块固定缓冲。这就是 **每步 ~6~7 次 `copy_`** 的来源——是 profile 里 `aten::copy_` 的大头。

> **关键启示**：multi-step（`tp_worker.py:604`）已经在干这件事的正确版本——它不调 `load_batch`，而是**直接写 `buffers.input_ids[0] = ...`、`buffers.positions[0] = ...`**（`tp_worker.py:664-668`），只改 bs=1 需要的那几个标量，然后 replay。这恰恰是"单步 decode 也该用的外科手术式更新"范式（详见优化 no.2）。

### 开销点 ③ overlap relay（future_map）：每步 2 次 dtype cast + scatter

代码栈 `overlap_utils.py:314/325`：
```python
# publish
self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)  # int32→int64 cast
# stash
self.output_tokens_buf[indices] = payload.to(torch.int64)                      # int32→int64 cast
```

大白话：overlap 用一块"全局池缓冲"把上一步采到的 token / 新 seq_len 传给下一步。每传一次都要 `.to(int64)`（因为采样出的是 int32 token，缓冲是 int64）。这是 profile 里 `aten::to` 的来源之一。每步 2~3 个小 kernel launch，单看不贵，架不住每步都发。

### 开销点 ④ `batch.copy()`：每步浅拷贝一个 ScheduleBatch 快照

代码栈 `scheduler.py:1552` + `schedule_batch.py:2840`：
```python
self.result_queue.append((batch.copy(), batch_result))   # scheduler.py:1552
# ScheduleBatch.copy() 只 self.reqs[:] 切片，其余 26 个字段按引用共享
```

大白话：overlap 把"上一步的结果处理"延后到"这一步"做，所以要把上一步的 batch 存个快照。好在这只是浅拷贝（张量都共享，不复制数据），开销是"构造一个新 Python 对象 + 切一个 list"。不大，但每步都做，属于"低垂的果子"。

### 开销点 ⑤ `next_token_ids.tolist()`：每步唯一的强制 GPU↔CPU 同步 ★

代码栈 `batch_result_processor.py:780`：
```python
next_token_ids = next_token_ids.tolist()   # next_token_ids 此时还在 GPU → 强制 D2H sync
```
配合 `:641 result.copy_done.synchronize()`。

大白话：DL 已经把 `copy_to_cpu` 跳过了（`scheduler.py:3216`，非 logprob decode 只录个空 event，token 留在 GPU 省一次 D2H）。但到了处理结果这一步，要把 token 变成 Python int 喂给 `req.output_ids.append(...)`，`.tolist()` 会**强制等 GPU 把这一步算完、把数据搬下来**——这就是"CPU 拽住 GPU"的那一下。这是结果处理里最大的单点同步。（multi-step 已用 `torch.stack([...]).tolist()` 把 N 个 token 合并成 1 次 sync，见 `batch_result_processor.py:690-698`。）

### 开销点 ⑥ output_streamer IPC：每步 pickle 一个 ~30 字段 dataclass 跨进程

代码栈 `output_streamer.py:117 / 517 / 163`：
```python
def _stream_output_generation(...):              # :117 每步调用
    payload = BatchTokenIDOutput(...)            # :517 to_payload，~30 个 list 字段
    self.send_to_detokenizer.send_output(payload) # :163 pickle + ZMQ 跨进程
```

大白话：sglang 把"调度/算"放 Scheduler 进程、"反 tokenize"放 Detokenizer 进程。每生成一个 token（stream_interval=1 时）都要把一个 30 字段的大对象序列化、走 ZMQ 发过去。这不是 `aten::copy_`，但实打实的 CPU 工作。vLLM 这块更紧凑。

---

## 4. "21 次 eager copy / 步"对号入座（归因表）

profile 实测 32 步：`aten::copy_` 673 次、`aten::to` 678 次、`aten::_to_copy` 261 次 ≈ **每步 21 次拷贝**。来源对号入座：

| 来源（每步） | 大致次数 | 对应开销点 |
|---|---|---|
| `load_batch`/`fill_from` 静态缓冲填充 | **~6~7 `copy_`** | ②（#1） |
| `prepare_for_decode` overlap 新张量（seq_lens/seq_lens_cpu/orig_seq_lens +1） | ~3 `add→new` | ① |
| overlap relay `publish`/`stash` dtype cast + scatter | ~2~3 `to` | ③ |
| `resolve_forward_inputs` 取上一步 token（index copy） | 1 `copy_` | ③ |
| `ForwardBatch.init_new` 的 H2D（global_num_tokens 等，若启用 MLP-sync） | 2~3 `to` | — |
| FP8 / MoE 每层 `.contiguous()` + `.to(dtype)`（`fp8_utils.py:523,568`） | 2×linears×layers | —（per-layer） |
| `next_token_ids.tolist()` 强制 D2H | 1 D2H sync | ⑤ |
| IPC pickle / `BatchTokenIDOutput` 构建 | 0 `aten`，纯 CPU | ⑥ |

**结论**：`load_batch` 的 6~7 次是绝对大头；其次 overlap 的 3 个新张量 + 2~3 次 dtype cast；再就是 FP8/MoE per-layer 的 cast（这部分在图里、被 replay 覆盖，host 主要是 launch 开销）。

> ⚠️ 注意：decode 走的是 **decode_cuda_graph_runner**（不是 eager runner）。`SGLANG_EAGER_INPUT_NO_COPY` 那个开关**只对 eager runner 生效**，CG runner 的 `load_batch` 走另一条路，无法用那个开关跳过。这是优化 no.2 的关键前提。

---

## 5. 30 个优化点（自问自答，no.1 ~ no.30）

> 每条格式：**问**（现状痛点）→ **答**（代码栈 + 大白话 + 做法 + 预期收益 + 风险）。
> 收益标注是**基于"差距 3.34ms、大头在 6ms/步 host 开销"的工程估计**，不是实测；带 ★ 的是优先做。

---

### no.1 ★ 用"带栈 profile"把 21 次拷贝逐个钉死在调用方

**问**：现在只知道"每步 ~21 次拷贝"，但不知道哪几次最贵、谁调用的。怎么先归因再动手？

**答**：
- 代码栈：抓 profile 时开 `SGLANG_PROFILE_WITH_STACK=true` + `SGLANG_TORCH_PROFILER_DIR=...`，跑 32 步 decode。
- 大白话：普通 profile 只告诉你"有 673 次 copy_"，带栈版本能告诉你"这 673 次里哪几次是 `fill_from` 调的、哪几次是 `stash` 调的"。先知道钱花在哪，再决定砍哪。
- 做法：抓两份（一份 overlap、一份 disable_overlap），对比拷贝来源差异。
- 预期收益：0（这是测量），但**决定后面 29 条的优先级**。
- 风险：profile 本身有开销，会扰动时序；只用于归因，不用于计时。

---

### no.2 ★★★ 消灭 `load_batch` 的 6~7 次静态缓冲拷贝（#1 元凶）

**问**：每步 decode 都要把 batch 整个 `copy_` 进图输入缓冲，能不能少拷？

**答**：
- 代码栈：`decode_cuda_graph_runner.py:866 load_batch` → `:876/912` → `fill_from`（`cuda_graph_buffer_registry.py:379`）。
- 大白话：CUDA graph 只认固定地址，所以"填缓冲"躲不掉。但 bs=1 时其实只需要改 **`input_ids[0]`、`positions[0]`、`seq_lens[0]`、`out_cache_loc[0]`** 这 4 个标量，`req_pool_indices` 单请求根本不变。`load_batch` 却把整批 slot 都 `foreach_copy_` 一遍。
- 做法：参考 multi-step（`tp_worker.py:664-668`）的写法，给 **bs=1 + 单请求 + overlap** 的常见情形加一条"外科手术式只写 4 个标量"的 fast-path，跳过 `fill_from` 的全量分桶拷贝。注意 `req_pool_indices` / `mrope_positions` 不变就跳过。
- 预期收益：**0.3~1ms/步**（6~7 次 foreach_copy_ → 4 次标量赋值）。这是单点最大的一笔。
- 风险：要保证图捕获时的静态缓冲布局与 fast-path 写入位置严格一致；换 batch size / 请求数时要回落到原 `load_batch`。需加正确性回归（greedy 输出逐 token 比对）。

---

### no.3 ★ `prepare_for_decode` overlap 新张量改成原地 / 预分配 ring

**问**：overlap 下每步 `seq_lens = seq_lens + 1` 造 3 个新张量（96 次/32 步），能不能不造？

**答**：
- 代码栈：`schedule_batch.py:2690-2697`。
- 大白话：造新张量是为了不和"正在跑的上一步 forward"抢同一块内存。但其实可以用**双缓冲 / 代际 ring**（上一步读 A、这一步写 B、轮换），原地 `add_` 即可，零新分配。
- 做法：给 `seq_lens` / `seq_lens_cpu` / `orig_seq_lens` 配 2 个交替缓冲，`prepare_for_decode` 里 `add_` 到"非在用"的那块，记录代际。或更简单：先实测 `add_` 原地是否真会 race（overlap 的 forward_stream 已 wait schedule_stream，见 `scheduler.py:1537`，可能本就安全）。
- 预期收益：**0.2~0.5ms/步**（省 3 次小张量分配 + 拷贝）。
- 风险：代际管理出 bug 会读到旧 seq_len → 错位。必须配正确性回归。

---

### no.4 统一 overlap relay 的 dtype，省掉每步 2~3 次 `.to(int64)` cast

**问**：`future_map.stash`/`publish` 每步都把 token / seq_len `int32→int64` cast 一次，能不能别 cast？

**答**：
- 代码栈：`overlap_utils.py:314`（`publish`）、`:325`（`stash`）。
- 大白话：采样出的 token 是 int32，全局池缓冲是 int64，每传一次都要 cast。如果让缓冲也用 int32（或采样直接出 int64），就省了 cast。
- 做法：把 `output_tokens_buf` / `new_seq_lens_buf` 改 int32；或让 sampler 在 greedy 路径直接 `.long()` 一次（在图里，便宜）。评估 cast 是不是真在 host 关键路径（很可能是 GPU 上随 replay 跑的小 kernel，host 只是 launch 开销）。
- 预期收益：**0.1~0.3ms/步**（主要是减少 launch 数，不是算力）。
- 风险：低；但需确认下游所有消费者都接受 int32。

---

### no.5 FP8 / MoE 每层 `.contiguous()` + `.to(dtype)`：bs=1 时能跳过吗

**问**：`fp8_utils.py:523 input.contiguous()` + `:568 out.to(dtype)` 每层每步都做，能不能砍？

**答**：
- 代码栈：`fp8_utils.py:523`（`input.reshape(...).contiguous()`）、`:568`（`out.to(dtype)`）、`:312-313`（blockwise scale `.contiguous()`）、`topk.py:1249/1281`（gating `.to(float32)`）。
- 大白话：这些 cast 大多在 **CUDA graph 内部**（捕获时就录进去了），replay 时不占 host——它们贡献的是"图里多几个小 kernel"，不是 host 拷贝。所以对 3.34ms host gap **基本无效**。但 `SGLANG_DL_GDN_BF16_BETA` 那类"去 beta cast"的实验已证明能省 ~0.3ms（噪声内），值得带栈 profile 确认哪几个 cast 真在图外。
- 做法：带栈 profile 筛出"图外的 cast"（即在 `load_batch`/`process_batch_result` 里的），只动那些；图内的别碰（动了改不了 host，反而要重捕图）。
- 预期收益：**0~0.3ms/步**（视图外 cast 占比而定）。
- 风险：乱删图内 cast 要重捕图，可能引入精度问题。**优先级低，放最后**。

---

### no.6 ★ `next_token_ids.tolist()`：异步化或进一步合并 sync

**问**：每步 `.tolist()` 是结果处理里唯一的强制 GPU 同步，能不能消掉？

**答**：
- 代码栈：`batch_result_processor.py:780`（`next_token_ids.tolist()`），`:641`（`copy_done.synchronize()`）。
- 大白话：DL 已经把 `copy_to_cpu` 跳了，token 留 GPU；但 `.tolist()` 又把它拽下来。这个 sync 是为了让 CPU 拿到 token 去 `req.output_ids.append()` 和算 finish。
- 做法：
  1. **路径 A**：恢复一次**异步** `copy_to_cpu`（`non_blocking=True` + event），让 `.tolist()` 在"下一步处理"时再 sync——把 sync 藏到 overlap 的下一轮，与 forward 重叠（vLLM 思路）。
  2. **路径 B**：bs=1 单请求时，finish 判定可以延后（多攒几个 token 再判 EOS），把 N 步合并成 1 次 sync（multi-step 已经在做，no.14 会深化）。
- 预期收益：**0.3~0.8ms/步**（消除每步的同步停顿，让 GPU 真正连续）。
- 风险：路径 A 要保证 overlap 下 token 生命期正确（别被下一步 `stash` 覆盖）；路径 B 会延迟 EOS 检�� 1~2 token，通常可接受。

---

### no.7 `batch.copy()` 快照：用代际号替代每步构造对象

**问**：每步 `batch.copy()` 浅拷贝一个 26 字段的 ScheduleBatch，能不能别造？

**答**：
- 代码栈：`scheduler.py:1552`（`self.result_queue.append((batch.copy(), batch_result))`）+ `schedule_batch.py:2840 copy()`。
- 大白话：浅拷贝其实不贵（张量共享），但每步构造一个 Python 对象 + 切一个 `reqs[:]`，累积有 GC 压力（见 no.12/no.26 的 27ms 尖峰嫌疑）。
- 做法：给 ScheduleBatch 加一个不可变的 `generation` 号，overlap 队列里存 `(generation, result)`，处理时按号取当前 batch 视图，不构造新对象。或者确认这个对象构造在 6ms 里占比 <1%，不值得动。
- 预期收益：**<0.1ms/步**（主要收益是降 GC，间接）。
- 风险：改动面大，收益小。**优先级低**。

---

### no.8 bs=1 单请求时 `process_batch_result_decode` 走 fast-path

**问**：`process_batch_result_decode` 的 per-req `for i, req in enumerate(...)` 循环，单请求时能不能跳过？

**答**：
- 代码栈：`batch_result_processor.py:675-754`（per-req 循环：`update_finish_state`、`time_stats`、`reasoning_tokens`、`free_group_begin/end`、`stream_output`）。
- 大白话：这个循环为多请求设计，bs=1 时一半分支（`is_spec`、`grammar`、`logprob`、`hidden_states`）都不会进，但 Python 解释器仍要逐个判断。
- 做法：加 `if len(batch.reqs) == 1 and not spec and not logprob:` 的特化分支，直接做最少必要工作（append token、判 finish、stream）。类似 multi-step 已有的"单请求特化"。
- 预期收益：**0.1~0.3ms/步**（省 Python 解释器开销）。
- 风险：低，但要维护两条路径一致性。

---

### no.9 `record_batch_in_overlap` 的 `attr_snapshot`：确认是否可关

**问**：每步 `record_batch_in_overlap` 构建 `attr_snapshot` 列表（`scheduler.py:3066/3073`），必要吗？

**答**：
- 代码栈：`scheduler.py:3066 record_batch_in_overlap` → `:3073 attr_snapshot = [...]`。
- 大白话：这是 overlap 为了"两步后恢复 ScheduleBatch 被在 forward 中改过的字段"做的快照。每步构建一个 Python list。
- 做法：先带栈 profile 看它占多少；若 <0.1ms 则别动（动了风险大于收益）。
- 预期收益：**0~0.1ms/步**。
- 风险：这个快照关乎正确性，乱关会坏 overlap 状态恢复。**仅测量，勿盲删**。

---

### no.10 bs=1 无 KV 回收时跳过 `free_group_begin/end`

**问**：`process_batch_result_decode` 每步调 `free_group_begin`/`free_group_end`（`:673/:755`），单请求不释放时能不能跳？

**答**：
- 代码栈：`batch_result_processor.py:673`（`token_to_kv_pool_allocator.free_group_begin()`）、`:755`（`free_group_end()`）。
- 大白话：这是批量释放 KV slot 的成组调用。decode 中途（请求没结束）根本不释放，但每步仍 begin/end 一遍。
- 做法：只在有请求 finished / retracted 时才进 free group；否则跳过。或测确认开销可忽略。
- 预期收益：**<0.1ms/步**。
- 风险：低。

---

### no.11 ★★ multi-step N=2 的噪声：`gc.disable()` + 预热收敛

**问**：multi-step N=2 单次能到 24.95ms（接近 vLLM 24ms），但噪声 ±1.5ms。怎么让它稳定？

**答**：
- 代码栈：`tp_worker.py:604 _dl_multi_step_decode`、`schedule_batch.py:2636`。
- 大白话：N=2 单次最优已经能打平 vLLM，问题是忽快忽慢。最大嫌疑是 Python GC 周期性停顿（生成期间对象多）和 GPU 降频。
- 做法：测两组——(a) 计时前 `gc.collect()` 预热 + `gc.disable()` 跑测； (b) `dlsmi -l` 锁定 GPU 频率。看 TPOT 标准差是否从 ±1.5ms 收敛到 ±0.3ms 内。
- 预期收益：**让 24.95ms 从"单次最佳"变"稳定中位数"** → 稳定打败 vLLM 的 24ms。
- 风险：`gc.disable()` 长跑会内存涨；serving 场景需周期性 `gc.collect()`。

---

### no.12 GPU 锁频 `dlsmi -l`：消除降频抖动

**问**：multi-step 噪声是不是 GPU 自己降频？

**答**：
- 大白话：GPU 有动态调频，负载低时降频、突加载时来不及升。decode 每步 20ms 之间 GPU 空闲，可能反复升降频 → 抖动。
- 做法：`dlsmi -l`（DLIN 版 nvidia-smi -lgc）锁定最大频率，跑 multi-step 看噪声。
- 预期收益：**0.3~1ms 抖动消除**（让 multi-step 可用）。
- 风险：锁频增功耗/发热；确认散热允许。

---

### no.13 ★ multi-step 循环里 `normal_decode_set_metadata` 同页跳过

**问**：multi-step 每步都调 `normal_decode_set_metadata` 重写整个 page_table（`tp_worker.py:674`），page_size=16 时需要吗？

**答**：
- 代码栈：`tp_worker.py:674 normal_decode_set_metadata(...)`。
- 大白话：page_size=16 意味着每 16 个 token 才跨一页。multi-step 连续 N 步内大概率在同一页，页表不变，每步重写是浪费。
- 做法：在循环里判断 `cur_seq_len % page_size != 0`（同页）时跳过 `set_metadata`，只在跨页那步更新。
- 预期收益：**0.2~0.5ms / N 步**（multi-step N=2~4 时显著）。
- 风险：低；跨页那步必须更新，边界要测准。

---

### no.14 multi-step 的 argmax：累积 N 步 logits 一次采样，或 fused sample

**问**：multi-step 每步在 GPU 上 `output.next_token_logits[0].argmax(dim=-1)`（`tp_worker.py:691`），vocab=151936，能不能省？

**答**：
- 代码栈：`tp_worker.py:691 next_id = output.next_token_logits[0].argmax(dim=-1, keepdim=True)`。
- 大白话：argmax 本身在图外、每步 1 次。N 步就 N 次。它本身不慢（GPU 上 ms 级以下），但每次都是 host 发射 + 等 GPU 出结果（多步的"metadata 依赖导致每步多 0.7ms"就是这，见报告 §6.2）。
- 做法：
  1. 把 argmax 也录进图（让 `next_token_logits` 直接喂给一个"采样子图"），multi-step 里只 replay 不取 logits。但这样拿不到 token 喂下一步 input_ids——除非用"上一步 token 索引"也做进图（graph-in-graph，复杂）。
  2. 退一步：N 步合并时，只对最后一步取 token 同步，中间步用 GPU 端 ring 自反馈（把 argmax 结果直接 `copy_` 进下一步 input_ids 缓冲，全程不上 host）。multi-step 的 `_dl_all_token_ids` 已经把 N 个 token 的 host 同步合并成 1 次（`batch_result_processor.py:690-698`），但循环内每步仍有 GPU→token 的依赖。
- 预期收益：**0.3~0.7ms / N 步**（消除每步的 metadata 同步停顿）。
- 风险：高。graph-in-graph 或 GPU 自反馈改动大，要保证 token 传递正确性。

---

### no.15 multi-step 的 0.7ms metadata 依赖：把 set_metadata 也做进图

**问**：报告 §6.2 说 multi-step 每 token replay 比单步多 0.7ms（metadata 依赖）。能不能消？

**答**：
- 大白话：这 0.7ms 是 `set_metadata`（更新页表/seq_len 缓冲）+ argmax 这两个"图外 GPU 工作"每步各发一遍的 launch + 同步开销。
- 做法：把 `normal_decode_set_metadata` 里那些**每步只 +1 的标量更新**（seq_len、position）改成直接写图内静态缓冲（multi-step 已经在写 `buffers.seq_lens[0]=cur_seq_len`，`tp_worker.py:668`），让它们随 replay 一起生效；argmax 按 no.14 处理。
- 预期收益：**0.5ms / N 步**。
- 风险：中；要保证图捕获时这些缓冲是可写的静态地址。

---

### no.16 greedy 时 lm_head + argmax 能否缩减候选集

**问**：argmax 在 151936 词表上扫，bs=1 时 lm_head 是 `[1,hidden]×[hidden,151936]` 大 GEMM，能不能只算 top 候选？

**答**：
- 大白话：greedy 只关心最大那个 token。理论上可以用"两阶段"（先粗筛 top-k 再精算），但 lm_head 这种 GEMM 在 GPU 上高度优化，分两阶段未必更快，且在 CG 内部。
- 做法：**GPU 侧（图内）优化，不归 host gap**。除非 DLIN 给 fused lm_head+argmax kernel，否则不动。归入"等 DLIN"一类。
- 预期收益：0（对 host gap 无效）。
- 风险：动 lm_head 改精度。**本条结论：不值得做**，列出仅为排除。

---

### no.17 把采样融合进 CUDA graph

**问**：sampler 现在在图外（`tp_worker.py:531 self.model_runner.sample(...)`），能不能录进图？

**答**：
- 大白话：vLLM 把采样做进图，replay 直接出 token。sglang 图外采样多一次 launch + logits 落地。
- 做法：把 `argmax`（greedy）录进图，输出 `next_token_ids` 到图输出缓冲。但这与 no.14 是同一件事的两面——做进图后，"下一步喂回 input_ids"就要靠 GPU 端自反馈（no.14）。两件事要一起做。
- 预期收益：**0.2~0.5ms/步**（省一次图外 launch + logits 同步）。
- 风险：高，与 no.14 耦合。

---

### no.18 bs=1 时 NCCL allreduce 的 algorithm tuning

**问**：TP4 下 MoE/dense 后有 allreduce，CG 内固定 algorithm。bs=1 时 ring 还是 tree 更快？

**答**：
- 代码栈：`sampler.py:260 _sync_token_ids_across_tp`（token 同步）+ MoE 后的 NCCL allreduce。
- 大白话：报告 §7.56 提到"NCCL ring vs tree"。bs=1 数据量极小，allreduce 启动延迟 > 传输时间，可能 ring 更优。但这是 **GPU 侧（图内）**，对 host gap 无直接帮助。
- 做法：仅在"等 DLIN"阶段顺带试 `NCCL_ALGO`/`NCCL_PROTO` 环境变量。
- 预期收益：**0~0.3ms GPU**（与 host gap 无关）。
- 风险：动 NCCL 配置可能影响稳定性。**优先级低**。

---

### no.19 ★★ IPC：把 detokenizer 与 scheduler 的"每步 pickle"换成增量 / 共享内存

**问**：output_streamer 每步 pickle 一个 ~30 字段 dataclass 走 ZMQ（`output_streamer.py:163`），能不能轻一点？

**答**：
- 代码栈：`output_streamer.py:117`（每步 `_stream_output_generation`）、`:517`（`to_payload` ~30 字段）、`:163`（`send_to_detokenizer.send_output`，pickle+ZMQ）。
- 大白话：vLLM 单进程没这开销；sglang 每步序列化一个 30 字段大对象。bs=1 单请求时，其实只要发"1 个 token id + rid"。
- 做法：
  1. **增量协议**：stream_interval=1 + 单请求时，只发 `{rid, token_id, finished}`，detokenizer 端自己维护 output_ids（它本来就有）。
  2. **共享内存环形缓冲**：scheduler 把 token 写进 shm ring，detokenizer 轮询/poll，零拷贝、零序列化。
- 预期收益：**0.3~0.8ms/步**（消除 pickle + 大对象构建）。
- 风险：中；改 IPC 协议要兼容多请求/多字段（logprob、reasoning）情形，回落要稳。

---

### no.20 `to_payload` 的 ~30 字段构建：bs=1 只发增量

**问**：`to_payload` 每步切 30 个 list（`output_streamer.py:517`），单请求能不能只切 1 个？

**答**：
- 大白话：和 no.19 是同一处的两面。no.19 改协议，no.20 改实现：构造 payload 时，无 logprob/无 reasoning/单请求时跳过那些空 list 字段的构建。
- 做法：`to_payload` 加 fast-path：`if not return_logprob and len(reqs)==1: 只填 rids/output_ids 尾 token`。
- 预期收益：**0.1~0.3ms/步**。
- 风险：低。

---

### no.21 stream_interval 配合 multi-step 批量 IPC

**问**：multi-step 一次产出 N 个 token（`req.output_ids.extend(_dl_ids)`，`batch_result_processor.py:699`），但 IPC 仍可能每步触发，能不能批量发？

**答**：
- 代码栈：`batch_result_processor.py:694-700`（multi-step 把 N token 一次 append 进 output_ids）+ `output_streamer.py` 的 `accept()` 按 stream_interval 门控。
- 大白话：multi-step 已经在"数据层"批量了，但 `stream_output` 的门控（`accept()` 里 `len(output_ids) % stream_interval`）若 stream_interval=1 仍每步发。可以让 multi-step 一次 IPC 发 N 个 token。
- 做法：multi-step 产出后，让 `accept()` 看到 output_ids 一次涨 N，按 stream_interval 决定发不发——天然就少 IPC。验证当前实现是否已经如此（很可能已经是，因为 extend 一次涨 N）。
- 预期收益：**0.1~0.3ms/步**（multi-step 模式下）。
- 风险：低；注意 streaming 体感（用户看到的 token 流间隔）。

---

### no.22 `alloc_for_decode` 每步调用：bs=1 预分配环形 slot 池

**问**：`prepare_for_decode` 每步 `alloc_for_decode(token_per_req=1)`（`schedule_batch.py:2638/2649`），能不能省？

**答**：
- 大白话：每步向 KV 池申请 1 个 slot。单请求长生成时，完全可以预申请一串连续 slot（环形），每步只移动指针。
- 做法：单请求 decode 时，生成开始预申请 `max_new_tokens` 个 slot，`prepare_for_decode` 只 `loc += 1`。multi-step 已经预申请 N 个（`schedule_batch.py:2640-2647`），把它推广到"预申请一整段"。
- 预期收益：**0.1~0.2ms/步**（省 allocator 调用 + 锁）。
- 风险：中；预申请占用显存（长生成时显存碎片），要与 retraction 配合。需上限保护。

---

### no.23 req_to_token 页表更新：同页跳过（与 no.13 呼应）

**问**：每步写 `req_to_token[req_pool_idx, cur_seq_len] = out_cache_loc`，page_size=16 时能否批量？

**答**：
- 大白话：跨页才需要"新页→slot"映射更新；同页内只是 slot 指针递增。multi-step 的 `buffers.out_cache_loc[0] = req_to_token[req_pool_idx, cur_seq_len]`（`tp_worker.py:664`）每步都查一次。
- 做法：同页时缓存上次 page_table，只递增 slot 索引。
- 预期收益：**<0.1ms/步**。
- 风险：低。**优先级低**，与 no.13 一起做。

---

### no.24 `ForwardBatch.init_new` 的每步列表推导：无 lora/无 rid-int 时跳过

**问**：`init_new` 每步 `[req.lora_id for req in batch.reqs]`、`[req.rid for req in batch.reqs]`（`forward_batch_info.py:710-711`），无 lora 时能不能跳？

**答**：
- 代码栈：`forward_batch_info.py:710-711`。
- 大白话：纯 Python list 构建，每步 2 次。无 lora 时 lora_id 全 0，仍要建 list。
- 做法：检测 `if not any(req.lora_id for req in batch.reqs): lora_ids = None`（下游已支持 None 时跳过）。
- 预期收益：**<0.05ms/步**。
- 风险：低。**极低优先级**，列出体现"连这种都想过"。

---

### no.25 ★ 测量修正：多进程下必须在 scheduler 进程内测 GPU

**问**：报告 §3.4 说 sglang 多进程下主进程 CUDA event 测量"不可靠"。怎么测才对？

**答**：
- 大白话：sglang 的 GPU 工作在 Scheduler 子进程，TokenizerManager 主进程拿不到准确的 CUDA event。`SGLANG_DL_TIME_REPLAY=1`（scheduler 进程内、正确 stream 上测）才是对的；主进程 event 只是"参考"。
- 做法：所有 GPU 计时统一走 `SGLANG_DL_TIME_REPLAY`；抓 vLLM 的对应 trace（`VLLM_TORCH_PROFILER_DIR`）在单进程下对比，确认 GPU forward 真等价。
- 预期收益：0（测量），但**堵住"是不是 GPU 其实慢"的质疑**。
- 风险：无。

---

### no.26 ★★ 定位周期性 27ms 尖峰：GC？NCCL？还是别的

**问**：报告 §3.3 说 wall-gap median 3.2ms 但有"周期性 27ms 尖峰"。这是啥？

**答**：
- 大白话：这是 sglang 比 vLLM 多出来的"偶发大卡顿"的元凶。嫌疑：(a) Python GC 分代回收停顿；(b) NCCL 内部 watch/超时；(c) CUDA caching allocator 大块回收；(d) OS 调度抢占。
- 做法：
  1. `py-spy dump --pid <scheduler_pid>` 在尖峰时刻抓栈，看是不是停在 `gc` / `zmq` / `nccl`。
  2. torch profiler 的 "python" + "memory" 视图看尖峰对齐谁。
  3. 实验：`gc.disable()` 跑、`PYTHONDONTWRITEBYTECODE`、调大 `gc` 阈值（`gc.set_threshold(50000)`）。
- 预期收益：**消除尖峰可能直接拿掉中位数的拖累**（若尖峰拉高了统计中位数，干掉它中位数可能直接掉 1~2ms）。
- 风险：中；GC 调参需长跑验证内存。

---

### no.27 内存分配器：减少每步小张量分配碎片

**问**：每步 3 个新 seq_lens 张量 + 各种小 buffer，allocator 压力大吗？

**答**：
- 大白话：torch CUDA caching allocator 对小张量友好（有缓存），但频繁 alloc/free 仍有锁和碎片。
- 做法：`torch.cuda.memory.set_per_process_memory_fraction` 预留 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减碎片。配合 no.3（少造新张量）一起。
- 预期收益：**0.1~0.3ms/步**（间接，主要降尖峰）。
- 风险：低。

---

### no.28 ★★（架构）把 decode 热路径从 Python 移到 C++/Cython

**问**：6ms/步 host 开销里有多少是"纯 Python 解释器"开销？能不能下沉？

**答**：
- 大白话：每步 event loop 的 `recv → prepare → run_batch → process → stream` 全是 Python，解释器逐行执行。vLLM 同类工作在 C++/tight loop 里。sglang 已有 **scripted runtime** 方向（把调度逻辑编译成固定脚本，减少 Python 分支）。
- 做法：查 sglang 的 scripted scheduler（`scripted_runtime_notes` skill），把 decode-only + 单请求 + overlap 的路径脚本化，跳过 Python 分支判断。这是 sglang 上游自己也在做的方向。
- 预期收益：**1~2ms/步**（理论上限最大的一笔，但工程量也最大）。
- 风险：高；scripted runtime 在 DLIN 上未验证（DLEOL JIT 状态敏感）。属于中长期。

---

### no.29 ★★★（架构）GPU-driven loop：persistent kernel / megakernel

**问**：vLLM host gap ≈ 0ms 的本质是什么？sglang 能复制吗？

**答**：
- 大白话：vLLM 让 CPU"发射一次大循环"后，GPU 自己连续生成多个 token（persistent / megakernel 思路，或图内多步），CPU 不参与每步。sglang 是"CPU 每步驱动一次 replay"。本质差距就在这。
- 做法：把 multi-step（no.11~15）推到极致——N 取大（8~32），且循环内**完全不上 host**（no.14 的 GPU 自反馈 + no.15 的图内 metadata），CPU 只在 N 步后回来一次。等价于"CPU 驱动一次、GPU 跑 N 步"。
- 预期收益：**2~3ms/步**（理论上能把 host 开销摊薄到接近 vLLM 的 3ms）。
- 风险：极高；纯 GPU 自反馈 loop 要保证 EOS 检测、请求进出、跨页 KV 正确。这是"真正打平 vLLM"的终局方案。

---

### no.30 ★（决策）投资回报率：要不要为这 3.34ms 做大改

**问**：sglang 已经到 vLLM 93%（§7.57），剩下 3.34ms 值得投入多少？

**答**：
- 大白话：把 30 条按"收益/风险/工程量"排个序，决定打哪些仗。
- 建议分三档：
  - **速赢（1~2 天，低风险，合计 ~1.5ms）**：no.1(归因) → no.2(load_batch fast-path) → no.3(seq_lens 原地) → no.6(.tolist 异步) → no.11/12(multi-step 稳定)。这组就能让 sglang 稳定进 25ms 内。
  - **中档（1 周，中风险，再 ~1ms）**：no.8(单请求 fast-path) → no.13/15(multi-step 图内) → no.19/20(IPC 增量) → no.26(尖峰定位)。
  - **终局（长线，高风险，打平 vLLM）**：no.14/17(采样入图 + GPU 自反馈) → no.28(scripted) → no.29(GPU-driven loop)。
- 决策：如果目标是"serving 对外打平 vLLM"，做速赢 + 中档即可（≈24ms）。如果目标是"极致延迟"，再上终局。
- 风险：终局改动大，可能与 sglang 上游演进冲突，需评估维护成本。

---

## 6. 优先级路线图（一图流）

```
                     收益 ↑
                      │
   no.29 GPU-driven   │   no.2 load_batch fast-path ★★★
   no.28 scripted     │   no.6 .tolist 异步 ★
   no.14/17 采样入图  │   no.3 seq_lens 原地 ★
                      │   no.11/12 multi-step 稳定 ★★
                      │   no.19 IPC 增量 ★★
                      │   no.26 尖峰定位 ★★
                      │
   ───────────────────┼────────────────────── 工程量/风险 →
   速赢档              │   中档                终局档
  (1~2天, ~1.5ms)      │  (1周, ~1ms)         (长线, 打平)
```

**明天第一步（确定性最高）**：no.1 带栈 profile 归因 → no.2 砍 `load_batch` 全量拷贝。这两步走完，3.34ms 里大概率先啃下 1ms，且风险最低。

---

## 7. 附录：测量口径备忘

- **Decode-only TPOT**（最干净）：prompt 极短（'Hi'），生成 512 token，TTFT 稀释到 ~0.4ms/tok，只看稳态。
- **纯 GPU forward**：`SGLANG_DL_TIME_REPLAY=1`，CUDA event 包 `graph.replay()` + `synchronize()`，在 **scheduler 进程内**测。
- **GPU 分解**：`SGLANG_DL_SKIP_MOE=1` 差分得 MoE/非 MoE 占比。
- **wall-gap between replays**：`SGLANG_DL_TIME_REPLAY=2`（无 sync），看 host 是否与 GPU 重叠 + 尖峰。
- **vLLM 对照**：`VLLM_TORCH_PROFILER_DIR` 抓 trace，单进程下 event 测（vLLM 主进程可信）。
- **eager copy 归因**：`SGLANG_PROFILE_WITH_STACK=true SGLANG_TORCH_PROFILER_DIR=...`，跑 32 步，把 `aten::copy_/to` 关联到 Python 调用方。

---

## 8. 一句话收尾

> GPU 内核已与 vLLM 完全相同、20.7ms 打平；3.34ms 差距 100% 在 sglang 每步 host 开销
> （#1 元凶是 `load_batch` 每步 6~7 次静态缓冲拷贝，其次是 overlap 新张量、`.tolist()` 同步、IPC pickle、周期性 27ms 尖峰）。
> 速赢档（no.1/2/3/6/11/12）就有望让 sglang 稳定进 25ms；终局档（no.14/17/28/29）才能把 host gap 压到 vLLM 的 ~0ms。
> **明天先做 no.1 带栈归因 + no.2 砍 load_batch**，这是性价比最高的第一步。
