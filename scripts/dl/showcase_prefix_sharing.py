#!/usr/bin/env python3
# DL: sglang vs vLLM prefix-sharing showcase benchmark.
#
# Tests RadixAttention (sglang) vs APC (vLLM) with:
#   Show Case 1: Multi-request prefix sharing — N requests sharing a LONG prefix
#   Show Case 2: Multi-turn conversation — each turn extends the previous
#   Show Case 3: Concurrent batch with shared prefix
#
# Engine: offline (sglang.Engine / vLLM LLM), sequential + concurrent.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/dl/showcase_prefix_sharing.py
#   CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm
import os, time, statistics, argparse

# DLIN env defaults
os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "32")
os.environ.setdefault("SGLANG_DL_GDN_DLIN", "1")
os.environ.setdefault("SGLANG_DL_MULTI_STEP", "1")
os.environ.setdefault("DLEOL_CACHE_SIZE", "1024")
os.environ.setdefault("DLEOL_FLA_ENABLE_PINGPONG", "1")
os.environ.setdefault("DLEOL_FLA_UNROLL_COUNT", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("DLEOL_USE_CU_MQA_TILEKV", "1")
os.environ.setdefault("VLLM_MAX_MOE_CU_TOKENS", "128")

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
TP = 4

# A long shared prefix (~2K tokens): system prompt + technical passage + few-shot examples
SHARED_PREFIX = """You are an expert data analyst. Answer questions about the following technical document precisely.

## Document: DLIN KS38 GPU Architecture Specification

The DLIN KS38 is a heterogeneous computing platform featuring 32 KS38 processing units organized as 8 boards, each containing 4 QUAD modules. Each QUAD operates independently with 32 GiB of local cluster memory. The interconnect topology uses a combination of PCIe Gen5 for host-device communication and custom high-bandwidth links for inter-board data transfer.

### Compute Capabilities
- FP32: 48 TFLOPS per QUAD
- BF16: 96 TFLOPS per QUAD
- FP16: 96 TFLOPS per QUAD
- INT8: 192 TOPS per QUAD
- FP8 (E4M3): 192 TFLOPS per QUAD

### Memory Hierarchy
The memory system consists of three tiers:
1. L1 Cache: 192 KiB per SM, 4-cycle latency
2. L2 Cache: 48 MiB shared, 20-cycle latency
3. Cluster Memory (HBM): 32 GiB per QUAD, 300-cycle latency, 1.2 TB/s bandwidth

### Programming Model
The KS38 supports CUDA-compatible programming through the DLCC compiler toolchain. Key extensions include:
- Programmatic Dependent Launch (PDL): Hopper-style dependent kernel launch for reducing launch overhead
- Dynamic Load Execution and Optimization Layer (DLEOL): runtime optimization framework with JIT compilation
- Custom Operations Library (_dl_C.so): pre-compiled fused kernels for MoE, attention, and quantization

### Performance Characteristics
For LLM inference workloads, the KS38 demonstrates:
- Decode throughput: 30-40 tok/s for 35B MoE models (FP8, TP4)
- Prefill latency: 200-600ms for 1K-token prompts
- KV cache efficiency: page-based management with 16-token page size
- MoE routing: 256 experts, top-8 selection, fused grouped GEMM

### Optimization Guidelines
1. Use FP8 quantization for both weights and activations (BF16 is 7x slower)
2. Enable cuda-graph capture for decode (saves ~10ms/token host overhead)
3. Set DLEOL_CACHE_SIZE=1024 for kernel JIT cache stability
4. Use chunked_prefill_size=16 for long prompts to avoid DLEOL crashes
5. Disable custom_all_reduce (use NCCL instead — DLIN HC_CUK Error=28)

### Benchmark Results
Measured on Qwen3.5-35B-A3B-FP8, TP4, 32 GiB cards:
- sglang decode: 35.0 tok/s (with seq_lens_sum optimization)
- vLLM MRV2 decode: 37.9 tok/s (FP8+cuda-graph)
- Gap: 6.2ms/token (100% host-side IPC overhead, GPU kernels identical)
- MoE kernel: invoke_fused_moe_opt, ~0.1ms/layer (fast path, no gather)
- Attention kernel: vllm_flash_attn varlen_fwd (both engines identical)

### Lessons Learned
1. RadixAttention provides token-level KV cache reuse for multi-turn and prefix-sharing workloads
2. Compressed FSM enables efficient structured output (JSON/regex) generation
3. The decode gap between sglang and vLLM is architectural (IPC), not kernel-level
4. torch.compile on DLIN requires Phase II fusion pass injection (not yet implemented)

## Few-shot Examples

Q: What is the FP8 throughput per QUAD? A: 192 TFLOPS per QUAD.

Q: How much L2 cache does the KS38 have? A: 48 MiB shared L2 cache with 20-cycle latency.

Q: What page size is used for KV cache management? A: 16-token page size.

Q: Why should FP8 be preferred over BF16 on KS38? A: BF16 is 7x slower than FP8 for decode workloads.

"""


def build_questions():
    """Build 8 questions that share SHARED_PREFIX."""
    return [
        "Q: What is the total FP8 throughput across all 32 QUADs? A:",
        "Q: How much total cluster memory does the KS38 have? A:",
        "Q: What is the recommended DLEOL_CACHE_SIZE setting? A:",
        "Q: Which all-reduce mode should be disabled on DLIN? A:",
        "Q: What is the L1 cache size per SM? A:",
        "Q: What compiler toolchain does KS38 use? A:",
        "Q: How many experts does the MoE routing use? A:",
        "Q: What is the HBM bandwidth per QUAD? A:",
    ]


def build_multi_turn():
    """Build a 5-turn conversation where each turn extends the previous."""
    turns = [
        "What is the FP8 throughput of a single QUAD on the KS38?",
        "And the total across all QUADs?",
        "How does that compare to the BF16 throughput?",
        "What is the memory bandwidth that supports this compute?",
        "Given all these specs, what is the expected decode throughput for a 35B model?",
    ]
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])
    parser.add_argument("--mem-frac", type=float, default=0.55)
    args = parser.parse_args()
    engine_name = args.engine

    print(f"[showcase] engine={engine_name} model={MODEL} tp={TP}", flush=True)

    # ---- Launch engine ----
    if engine_name == "sglang":
        import sglang as sgl
        engine = sgl.Engine(
            model_path=MODEL, tp_size=TP, dtype="bfloat16",
            context_length=4096, mem_fraction_static=args.mem_frac,
            max_running_requests=4, disable_cuda_graph=False,
            cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
            disable_custom_all_reduce=True, trust_remote_code=True,
            chunked_prefill_size=512,
        )
        def generate(prompt, max_new=32, temperature=0.0):
            return engine.generate(prompt, {"max_new_tokens": max_new, "temperature": temperature})
        def generate_batch(prompts, max_new=32, temperature=0.0):
            return engine.generate(prompts, {"max_new_tokens": max_new, "temperature": temperature})
        def shutdown():
            engine.shutdown()
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
            max_model_len=4096, gpu_memory_utilization=args.mem_frac,
            trust_remote_code=True, enforce_eager=False, max_num_seqs=4,
            disable_log_stats=True,
            compilation_config={"cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4},
        )
        def generate(prompt, max_new=32, temperature=0.0):
            sp = SamplingParams(temperature=temperature, max_tokens=max_new)
            out = llm.generate([prompt], sp)[0]
            return out.outputs[0].text
        def generate_batch(prompts, max_new=32, temperature=0.0):
            sp = SamplingParams(temperature=temperature, max_tokens=max_new)
            outs = llm.generate(prompts, sp)
            return [o.outputs[0].text for o in outs]
        def shutdown():
            pass

    print(f"[showcase] {engine_name} engine up", flush=True)

    # ---- Warmup ----
    for _ in range(3):
        generate("Hello world", max_new=16)
    print(f"[showcase] {engine_name} warmup done", flush=True)

    questions = build_questions()
    turns = build_multi_turn()

    # ================================================================
    # SHOW CASE 1: Multi-request prefix sharing
    # 8 requests share a 2K-token prefix, each with different 10-token suffix.
    # Sequential: req1=cold, req2-8=warm (prefix cached).
    # Measure: cold time, warm time, speedup.
    # ================================================================
    print(f"\n[showcase] === SC1: Prefix Sharing (8 reqs, 2K shared prefix) ===", flush=True)

    # Cold: first request with this prefix
    cold_prompt = SHARED_PREFIX + "\n" + questions[0]
    t0 = time.perf_counter()
    cold_out = generate(cold_prompt, max_new=32)
    cold_time = time.perf_counter() - t0
    if isinstance(cold_out, dict):
        cold_out = cold_out.get("text", str(cold_out))

    # Warm: subsequent requests with SAME prefix + different suffix
    warm_times = []
    warm_outs = []
    for q in questions[1:]:
        warm_prompt = SHARED_PREFIX + "\n" + q
        t0 = time.perf_counter()
        out = generate(warm_prompt, max_new=32)
        dt = time.perf_counter() - t0
        warm_times.append(dt)
        if isinstance(out, dict):
            out = out.get("text", str(out))
        warm_outs.append(out)

    warm_med = statistics.median(warm_times)
    warm_min = min(warm_times)
    speedup = cold_time / warm_med

    print(f"[showcase] SC1 {engine_name}: cold={cold_time*1000:.0f}ms  "
          f"warm_median={warm_med*1000:.0f}ms  warm_min={warm_min*1000:.0f}ms  "
          f"speedup={speedup:.1f}x", flush=True)
    if isinstance(cold_out, str):
        print(f"[showcase] SC1 {engine_name} cold_sample: {cold_out[:80]!r}", flush=True)
    if warm_outs and isinstance(warm_outs[0], str):
        print(f"[showcase] SC1 {engine_name} warm_sample: {warm_outs[0][:80]!r}", flush=True)

    # ================================================================
    # SHOW CASE 2: Multi-turn conversation TTFT
    # 5 turns, each extends the conversation history.
    # Measure: per-turn wall time (should decrease as more is cached).
    # ================================================================
    print(f"\n[showcase] === SC2: Multi-turn Conversation (5 turns) ===", flush=True)

    conversation = SHARED_PREFIX  # Start with system prompt
    turn_times = []
    for i, turn_q in enumerate(turns):
        conversation += f"\n\nHuman: {turn_q}\nAssistant:"
        t0 = time.perf_counter()
        out = generate(conversation, max_new=32)
        dt = time.perf_counter() - t0
        turn_times.append(dt)
        if isinstance(out, dict):
            out = out.get("text", str(out))
        conversation += f" {out}"
        print(f"[showcase] SC2 {engine_name} turn{i+1}: {dt*1000:.0f}ms  "
              f"prompt_len={len(conversation)}  out={str(out)[:50]!r}", flush=True)

    # ================================================================
    # SHOW CASE 3: Concurrent batch with shared prefix
    # 4 requests with SAME prefix, sent as a BATCH.
    # Both engines should dedup the prefix internally.
    # Measure: total batch time, per-request throughput.
    # ================================================================
    print(f"\n[showcase] === SC3: Concurrent Batch (4 reqs, shared prefix) ===", flush=True)

    batch_prompts = [SHARED_PREFIX + "\n" + q for q in questions[:4]]
    # Run 3 times, take best
    best_batch = 999
    for _ in range(3):
        t0 = time.perf_counter()
        batch_outs = generate_batch(batch_prompts, max_new=32)
        dt = time.perf_counter() - t0
        best_batch = min(best_batch, dt)

    total_tokens = 4 * 32  # 4 reqs × 32 output tokens each
    batch_tps = total_tokens / best_batch
    print(f"[showcase] SC3 {engine_name}: batch_time={best_batch*1000:.0f}ms  "
          f"total_tokens={total_tokens}  throughput={batch_tps:.1f} tok/s  "
          f"per_req={best_batch*1000/4:.0f}ms", flush=True)

    # ================================================================
    # SHOW CASE 4: JSON structured output (both engines)
    # ================================================================
    print(f"\n[showcase] === SC4: JSON Structured Output ===", flush=True)

    json_prompt = ("Extract the person info as JSON: Dr. Ada Lovelace, 36, "
                   "senior research scientist at DeepMind in London. "
                   "Email: ada.l@deepmind.example\n\nJSON:")
    import json as _json
    json_schema = _json.dumps({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "occupation", "city"],
    })

    if engine_name == "sglang":
        # sglang with json_schema constraint
        best_json = 999
        json_valid = False
        json_sample = ""
        for _ in range(3):
            t0 = time.perf_counter()
            out = engine.generate(json_prompt, {
                "max_new_tokens": 48, "temperature": 0, "ignore_eos": True,
                "json_schema": json_schema,
            })
            dt = time.perf_counter() - t0
            best_json = min(best_json, dt)
            json_sample = out["text"][:120] if isinstance(out, dict) else str(out)[:120]
            try:
                _json.loads(out["text"] if isinstance(out, dict) else out)
                json_valid = True
            except Exception:
                pass
        json_tps = 48 / best_json
        print(f"[showcase] SC4 {engine_name}: time={best_json*1000:.0f}ms  "
              f"tok/s={json_tps:.1f}  valid={json_valid}", flush=True)
        print(f"[showcase] SC4 {engine_name} sample: {json_sample!r}", flush=True)
    else:
        from vllm.sampling_params import StructuredOutputsParams
        sp_json = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        sp_json.structured_outputs = StructuredOutputsParams(json=json_schema)
        best_json = 999
        json_valid = False
        json_sample = ""
        for _ in range(3):
            t0 = time.perf_counter()
            out = llm.generate([json_prompt], sp_json)[0]
            dt = time.perf_counter() - t0
            best_json = min(best_json, dt)
            txt = out.outputs[0].text
            json_sample = txt[:120]
            try:
                _json.loads(txt)
                json_valid = True
            except Exception:
                pass
        json_tps = 48 / best_json
        print(f"[showcase] SC4 {engine_name}: time={best_json*1000:.0f}ms  "
              f"tok/s={json_tps:.1f}  valid={json_valid}", flush=True)
        print(f"[showcase] SC4 {engine_name} sample: {json_sample!r}", flush=True)

    # ================================================================
    print(f"\n[showcase] === {engine_name} SUMMARY ===", flush=True)
    print(f"  SC1 cold={cold_time*1000:.0f}ms warm={warm_med*1000:.0f}ms speedup={speedup:.1f}x", flush=True)
    print(f"  SC2 turns={[f'{t*1000:.0f}' for t in turn_times]} ms", flush=True)
    print(f"  SC3 batch={best_batch*1000:.0f}ms throughput={batch_tps:.1f} tok/s", flush=True)
    print(f"  SC4 json={best_json*1000:.0f}ms tok/s={json_tps:.1f} valid={json_valid}", flush=True)

    shutdown()


if __name__ == "__main__":
    main()
