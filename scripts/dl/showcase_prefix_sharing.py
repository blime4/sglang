#!/usr/bin/env python3
# DL: sglang vs vLLM prefix-sharing showcase benchmark.
#
# Tests RadixAttention (sglang) vs APC (vLLM) with:
#   Show Case 1: Multi-request prefix sharing — N requests sharing a LONG prefix
#   Show Case 2: Multi-turn conversation — each turn extends the previous
#   Show Case 3: Concurrent batch with shared prefix
#   Show Case 4: JSON structured output
#
# Engine: offline (sglang.Engine / vLLM LLM), sequential + concurrent.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/dl/showcase_prefix_sharing.py
#   CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/dl/showcase_prefix_sharing.py --engine vllm
#   # only some scenarios (SC1 cold is ~90s; skip it for a fast SC2/SC3 check):
#   .venv/bin/python scripts/dl/showcase_prefix_sharing.py --scenarios SC2,SC3
#
# Or via run_sglang.sh:  ./run_sglang.sh compare [--scenarios SC1,SC2,SC3]
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
# NOTE: VLLM_USE_V2_MODEL_RUNNER is set explicitly at engine-launch time from
# --vllm-runner (mrv2 -> "1", mrv1 -> "0"); do NOT setdefault it here.
os.environ.setdefault("DLEOL_USE_CU_MQA_TILEKV", "1")
os.environ.setdefault("VLLM_MAX_MOE_CU_TOKENS", "128")

MODEL = os.environ.get(
    "MODEL_PATH",
    "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8",
)
TP = int(os.environ.get("TP_SIZE", "4"))

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


def build_long_doc(target_words=1150):
    """DL: build a ~2K-token shared document (RAG-scale prefix).

    SHARED_PREFIX (~700 tokens) + a repeated dense paragraph grown to
    ~target_words. A LONG prefix makes re-prefill (vLLM APC-off) expensive,
    which isolates the RadixAttention cross-request reuse win at RAG scale —
    the whole point of SC7. Repeated prose is fine: SC7 measures prefill
    compute cost, not semantics (output is ignore_eos fixed-length).
    """
    para = ("The KS38 heterogeneous accelerator combines 32 processing units "
            "across 8 boards with 4 QUAD modules each, delivering 192 FP8 "
            "TFLOPS per QUAD and 1.2 TB/s HBM bandwidth, programmed via the "
            "DLCC toolchain with PDL dependent-launch and the DLEOL JIT layer. ")
    body = SHARED_PREFIX + "\n## Extended Technical Notes\n\n"
    while len(body.split()) < target_words:
        body += para
    return body


def build_fork_users():
    """DL: two independent 4-turn conversations sharing a common root prefix.

    Exercises a 2-branch radix tree (the structure RadixAttention is named
    for): both users share the system/context root, then each grows its OWN
    branch. Distinct from SC2 (one linear conversation) and SC1 (flat prefix,
    unrelated questions). Production shape: multiple users / agents behind a
    shared system prompt.
    """
    user_a = [
        "What is the FP8 throughput of a single QUAD on the KS38?",
        "And the total across all 32 QUADs?",
        "How much HBM bandwidth supports that compute?",
        "Summarize the compute capability in one sentence.",
    ]
    user_b = [
        "What compiler toolchain does the KS38 use?",
        "What is DLEOL and what does it do?",
        "Why must custom_all_reduce be disabled on DLIN?",
        "What DLEOL_CACHE_SIZE is recommended and why?",
    ]
    return user_a, user_b


def build_rag_questions():
    """DL: 16 diverse short questions over the long shared doc (SC7)."""
    return [
        "What is the FP8 throughput per QUAD?",
        "How much L2 cache does the KS38 have?",
        "What is the HBM bandwidth per QUAD?",
        "What page size is used for KV cache management?",
        "Why prefer FP8 over BF16 on KS38?",
        "What is DLEOL_CACHE_SIZE recommended value?",
        "Which all-reduce mode should be disabled?",
        "What is the L1 cache size per SM?",
        "What compiler toolchain does KS38 use?",
        "How many experts does the MoE routing use?",
        "What is the BF16 throughput per QUAD?",
        "How many QUADs are on each board?",
        "What interconnect is used for host-device communication?",
        "What is the decode throughput for a 35B MoE model?",
        "What does PDL stand for?",
        "How much cluster memory per QUAD?",
    ]


def build_tenant_questions():
    """DL: 24 diverse short questions for the shared system-prompt scenario (SC10).

    All share the SAME ~0.9K-token system/context prefix (SHARED_PREFIX) — the
    canonical production RadixAttention shape: one chatbot/agent system prompt
    served to many users. Distinct from SC7 (3K doc RAG) by short prefix +
    higher request count.
    """
    return [
        "What is the FP8 throughput per QUAD?",
        "How much L2 cache does the KS38 have?",
        "What is the HBM bandwidth per QUAD?",
        "What page size is used for KV cache management?",
        "Why prefer FP8 over BF16 on KS38?",
        "What is the recommended DLEOL_CACHE_SIZE?",
        "Which all-reduce mode should be disabled?",
        "What is the L1 cache size per SM?",
        "What compiler toolchain does KS38 use?",
        "How many experts does the MoE routing use?",
        "What is the BF16 throughput per QUAD?",
        "How many QUADs are on each board?",
        "What interconnect is used host-device?",
        "What is the decode throughput for a 35B MoE model?",
        "What does PDL stand for?",
        "How much cluster memory per QUAD?",
        "How many processing units does the KS38 have?",
        "How many boards are in the KS38?",
        "What is the FP32 throughput per QUAD?",
        "What is the L2 cache latency in cycles?",
        "What is the INT8 throughput per QUAD?",
        "What is the cluster memory latency in cycles?",
        "What does DLEOL stand for?",
        "What is the HBM latency in cycles?",
    ]


def build_sc11_queries():
    """DL: 12 user queries of DELIBERATELY DIFFERENT lengths for SC11 (online
    concurrency). With the shared SHARED_PREFIX system prompt, the concurrent
    batch has UNEQUAL extend_lens -> the FA2 varlen path that fix 0fe8c7cc86
    unblocked (pre-fix: >2 overlapping unequal prefills OOB/SIGSEGV). Short
    factual prompts + a few long ones -> a clear length spread."""
    return [
        "What is the FP8 throughput per QUAD?",
        "L2 cache?",
        "Explain in detail the three-tier memory hierarchy of the KS38: the L1, L2, "
        "and cluster memory (HBM), including their sizes, access latencies in cycles, "
        "and bandwidth, and how a kernel author should reason about data placement.",
        "HBM bandwidth per QUAD?",
        "What page size is used for KV cache management, and why does it matter for "
        "paged attention efficiency and memory fragmentation?",
        "DLEOL_CACHE_SIZE?",
        "Describe the MoE routing: number of experts, top-k selection, and the fused "
        "grouped GEMM kernel, and how it differs from a dense MLP forward.",
        "Why must custom_all_reduce be disabled on DLIN, what mode is used instead, "
        "and what was the error code?",
        "INT8 throughput?",
        "Explain PDL (Programmatic Dependent Launch) and DLEOL (the JIT optimization "
        "layer): what each does, how they interact, and when each helps latency.",
        "Cluster memory latency in cycles?",
        "Summarize the KS38 compute capabilities across all numeric formats (FP32, "
        "BF16, FP16, INT8, FP8) per QUAD and in aggregate, with the tradeoffs.",
    ]


def build_unique_prompt(i, target_words=700):
    """DL: a ~1K-token prompt with a UNIQUE prefix per i (no shared root across
    different i) — for raw-prefill testing where RadixAttention must NOT hit.
    The distinct header (different first tokens per i) defeats root-matching.
    """
    header = f"### Analysis record {i} (run id {i * 7919 % 10007}): "
    seed = (f"in record {i} the measured metric {i} for component {i} "
            f"yields value {i} under load step {i}, which on device {i} ")
    body = header
    while len(body.split()) < target_words:
        body += seed
    return body


# ---------------------------------------------------------------------------
# Individual showcase scenarios. Each returns a metrics dict (KEY=value, all
# lower-level numbers are floats/ints) so run_sglang.sh's `compare` phase can
# parse them via "^METRIC KEY=" lines and build a side-by-side table.
# ---------------------------------------------------------------------------

def run_sc1(engine_name, generate, questions):
    """SC1: Multi-request prefix sharing. 8 reqs share a 2K prefix."""
    print(f"\n[showcase] === SC1: Prefix Sharing (8 reqs, 2K shared prefix) ===", flush=True)
    cold_prompt = SHARED_PREFIX + "\n" + questions[0]
    t0 = time.perf_counter()
    cold_out = generate(cold_prompt, max_new=32, ignore_eos=True)
    cold_time = time.perf_counter() - t0
    if isinstance(cold_out, dict):
        cold_out = cold_out.get("text", str(cold_out))

    warm_times, warm_outs = [], []
    for q in questions[1:]:
        t0 = time.perf_counter()
        out = generate(SHARED_PREFIX + "\n" + q, max_new=32, ignore_eos=True)
        warm_times.append(time.perf_counter() - t0)
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
    return {"SC1_cold_ms": f"{cold_time*1000:.0f}",
            "SC1_warm_ms": f"{warm_med*1000:.0f}",
            "SC1_speedup_x": f"{speedup:.2f}"}


def run_sc2(engine_name, generate, turns):
    """SC2: Multi-turn conversation. 5 turns, each extends history."""
    print(f"\n[showcase] === SC2: Multi-turn Conversation (5 turns) ===", flush=True)
    conversation = SHARED_PREFIX
    turn_times = []
    for i, turn_q in enumerate(turns):
        conversation += f"\n\nHuman: {turn_q}\nAssistant:"
        t0 = time.perf_counter()
        out = generate(conversation, max_new=32, ignore_eos=True)
        dt = time.perf_counter() - t0
        turn_times.append(dt)
        if isinstance(out, dict):
            out = out.get("text", str(out))
        conversation += f" {out}"
        print(f"[showcase] SC2 {engine_name} turn{i+1}: {dt*1000:.0f}ms  "
              f"prompt_len={len(conversation)}  out={str(out)[:50]!r}", flush=True)
    avg_ms = statistics.mean(turn_times) * 1000
    turn5_ms = turn_times[-1] * 1000
    print(f"[showcase] SC2 {engine_name}: avg={avg_ms:.0f}ms turn5={turn5_ms:.0f}ms "
          f"trend={[f'{t*1000:.0f}' for t in turn_times]} ms", flush=True)
    return {"SC2_avg_ms": f"{avg_ms:.0f}", "SC2_turn5_ms": f"{turn5_ms:.0f}"}


def run_sc3(engine_name, generate_batch, questions):
    """SC3: Concurrent batch with shared prefix. 4 reqs as a batch."""
    print(f"\n[showcase] === SC3: Concurrent Batch (4 reqs, shared prefix) ===", flush=True)
    batch_prompts = [SHARED_PREFIX + "\n" + q for q in questions[:4]]
    best_batch = 999
    for _ in range(int(os.environ.get("SHOWCASE_REPS", "3"))):
        t0 = time.perf_counter()
        generate_batch(batch_prompts, max_new=32)
        best_batch = min(best_batch, time.perf_counter() - t0)
    total_tokens = 4 * 32
    batch_tps = total_tokens / best_batch
    per_req_ms = best_batch * 1000 / 4
    print(f"[showcase] SC3 {engine_name}: batch_time={best_batch*1000:.0f}ms  "
          f"total_tokens={total_tokens}  throughput={batch_tps:.1f} tok/s  "
          f"per_req={per_req_ms:.0f}ms", flush=True)
    return {"SC3_throughput_tps": f"{batch_tps:.1f}",
            "SC3_per_req_ms": f"{per_req_ms:.0f}"}


def run_sc4(engine_name, raw, json_schema):
    """SC4: JSON structured output (warmed). raw = sglang.Engine or vLLM LLM."""
    print(f"\n[showcase] === SC4: JSON Structured Output ===", flush=True)
    json_prompt = ("Extract the person info as JSON: Dr. Ada Lovelace, 36, "
                   "senior research scientist at DeepMind in London. "
                   "Email: ada.l@deepmind.example\n\nJSON:")
    import json as _json
    best_json, json_valid, json_sample = 999, False, ""

    if engine_name == "sglang":
        # Warm the schema compiler first (xgrammar FSM build is ~3s one-time per
        # schema; without this warmup every request re-pays it and tok/s looks
        # 3x worse than steady state). vLLM caches its schema in the backend.
        for _ in range(2):
            raw.generate(json_prompt, {"max_new_tokens": 8, "temperature": 0,
                                        "ignore_eos": True, "json_schema": json_schema})
        for _ in range(3):
            t0 = time.perf_counter()
            out = raw.generate(json_prompt, {"max_new_tokens": 48, "temperature": 0,
                                              "ignore_eos": True, "json_schema": json_schema})
            best_json = min(best_json, time.perf_counter() - t0)
            txt = out["text"] if isinstance(out, dict) else str(out)
            json_sample = txt[:120]
            try:
                _json.loads(out["text"] if isinstance(out, dict) else out)
                json_valid = True
            except Exception:
                pass
    else:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        sp_json = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        sp_json.structured_outputs = StructuredOutputsParams(json=json_schema)
        sp_warm = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        sp_warm.structured_outputs = StructuredOutputsParams(json=json_schema)
        for _ in range(2):
            raw.generate([json_prompt], sp_warm)
        for _ in range(3):
            t0 = time.perf_counter()
            out = raw.generate([json_prompt], sp_json)[0]
            best_json = min(best_json, time.perf_counter() - t0)
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
    return {"SC4_tps": f"{json_tps:.1f}", "SC4_valid": "1" if json_valid else "0"}


def run_sc5(engine_name, generate):
    """DL: SC5 — Multi-user fork (2-branch radix tree).

    Two users each hold a 4-turn conversation over a SHARED system/context
    root, interleaved. RadixAttention caches the root once and each user's
    growing branch; vLLM (APC unsupported on this hybrid Mamba) re-prefills
    the whole conversation every turn. Distinct from SC2 (one linear convo)
    and SC1 (flat prefix, unrelated questions) — this is a branching tree,
    the structure RadixAttention is named for. Production shape: N users /
    agents behind one system prompt.
    """
    print(f"\n[showcase] === SC5: Multi-user Fork (2 users x 4 turns, shared root) ===", flush=True)
    user_a, user_b = build_fork_users()
    # Interleave so both branches grow from the shared root in one pass.
    order = [("A", user_a[0]), ("B", user_b[0]),
             ("A", user_a[1]), ("B", user_b[1]),
             ("A", user_a[2]), ("B", user_b[2]),
             ("A", user_a[3]), ("B", user_b[3])]

    def _txt(out):
        if isinstance(out, dict):
            t = out.get("text", str(out))
            return t[0] if isinstance(t, list) else t
        if isinstance(out, list):
            return out[0]
        return str(out)

    def one_pass():
        hist = {"A": SHARED_PREFIX, "B": SHARED_PREFIX}
        t0 = time.perf_counter()
        for who, q in order:
            prompt = hist[who] + f"\n\nUser: {q}\nAssistant:"
            out = generate(prompt, max_new=24, ignore_eos=True)
            hist[who] = prompt + " " + _txt(out)
        return time.perf_counter() - t0

    one_pass()  # warmup: populate the radix tree for both branches
    times = [one_pass() for _ in range(int(os.environ.get("SHOWCASE_REPS", "2")))]
    best = min(times)
    avg_turn = best / 8 * 1000
    print(f"[showcase] SC5 {engine_name}: best_total={best*1000:.0f}ms  "
          f"avg_turn={avg_turn:.0f}ms  reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)
    return {"SC5_total_ms": f"{best*1000:.0f}", "SC5_avg_turn_ms": f"{avg_turn:.0f}"}


def run_sc7(engine_name, generate):
    """DL: SC7 — Long-prefix RAG throughput (~2K shared doc, 8 queries).

    A long shared document (~2K tokens) + 8 diverse short questions, sent as
    SEQUENTIAL requests. RadixAttention prefills the doc once and each query
    only extends its short suffix; vLLM (APC-off) re-prefills the full doc on
    all 8. Reported as aggregate decode throughput (best-of-2 after warmup),
    which isolates the cross-request reuse win at RAG scale — bigger prefix
    and throughput-focused vs SC1 (per-request latency, 8 reqs, ~0.9K prefix).
    """
    print(f"\n[showcase] === SC7: Long-RAG Throughput (~2K shared doc, 8 queries) ===", flush=True)
    doc = build_long_doc()
    questions = build_rag_questions()[:8]   # 8 queries (vLLM re-prefills the doc each)
    ntok = len(doc.split())  # ~word count, logged for reference
    total_decode = len(questions) * 24

    def one_pass():
        t0 = time.perf_counter()
        for q in questions:
            generate(doc + "\n\nQ: " + q + "\nA:", max_new=24, ignore_eos=True)
        return time.perf_counter() - t0

    one_pass()  # warmup: cache the long doc once
    times = [one_pass() for _ in range(int(os.environ.get("SHOWCASE_REPS", "2")))]
    best = min(times)
    tps = total_decode / best
    print(f"[showcase] SC7 {engine_name}: doc~{ntok}words  best={best*1000:.0f}ms  "
          f"throughput={tps:.1f} tok/s  reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)
    return {"SC7_throughput_tps": f"{tps:.1f}", "SC7_total_ms": f"{best*1000:.0f}"}


def run_sc8(engine_name, generate):
    """DL: SC8 — Repeated best-of-N (RLHF rejection-sampling loop), n=4, temp=0.7.

    The SAME ~0.9K prompt is regenerated n=4 across warmup+measured reps, so
    sglang's RadixAttention caches the prompt across calls (reps skip prefill)
    while vLLM (APC-off) re-prefills it every call. This is the *repeated*
    best-of-N (RLHF-loop) advantage, NOT a single best-of-N call. Rigor
    decomposition (SC8b + docs/dl/sglang-vs-vllm-rigor-analysis.md): the win =
    cross-call caching (~3.9x) x a single-call best-of-N edge (~2x; vLLM's n=4
    decode is pathologically slow on DLIN). For the cold single-call best-of-N
    figure, see SC8b. (sglang's RAW prefill is NOT faster than vLLM's — see SC6.)
    """
    print(f"\n[showcase] === SC8: Repeated best-of-N (RLHF loop), n=4, temp=0.7 ===", flush=True)
    prompt = (SHARED_PREFIX + "\n\nWrite a concise technical summary of the "
              "KS38 architecture, compute, and memory hierarchy.\n\nSummary:")
    max_new = 48

    def _samples(out):
        # Normalize all engines' n>1 returns to a list[str]:
        #   sglang -> [{"text": s1,...}, ...]  (list of dicts)  [offline Engine n>1]
        #          OR {"text": [s1,...,s4]}    (dict of list)
        #   vLLM   -> [s1, s2, s3, s4]         (list of str)
        def _str(x):
            if isinstance(x, dict):
                return x.get("text", str(x))
            return str(x)
        if isinstance(out, dict):
            t = out.get("text", "")
            if isinstance(t, list):
                return [_str(x) for x in t]
            return [str(t)]
        if isinstance(out, list):
            return [_str(x) for x in out]
        return [str(out)]

    for _ in range(2):  # warmup (best-of-N needs sampling, NOT greedy)
        generate(prompt, max_new=max_new, ignore_eos=True, n=4, temperature=0.7)
    best, samples = 999, []
    for _ in range(3):
        t0 = time.perf_counter()
        outs = generate(prompt, max_new=max_new, ignore_eos=True, n=4, temperature=0.7)
        best = min(best, time.perf_counter() - t0)
        samples = _samples(outs)
    tps = (4 * max_new) / best
    sample = (samples[0][:60] if samples else "")
    print(f"[showcase] SC8 {engine_name}: best={best*1000:.0f}ms  tok/s={tps:.1f}  "
          f"n=4 x {max_new}tok  n_samples={len(samples)}  sample={sample!r}", flush=True)
    return {"SC8_tps": f"{tps:.1f}"}


def run_sc8b(engine_name, generate):
    """DL: SC8b — best-of-N, COLD (unique prompt each call) — rigor control.

    Identical to SC8 EXCEPT a UNIQUE ~0.9K prompt per measured call, so
    RadixAttention gets NO cross-call cache hit. This is the FAIR single-call
    best-of-N: both engines prefill the prompt once per call, then fork n=4.
    Compare to SC8 (same prompt reused => sglang caches across reps) to isolate
    that confound. If SC8b shows vLLM >= sglang (decode-bound), then SC8's
    sglang win was the cross-call caching, NOT best-of-N speed — and SC8 should
    be framed as a repeated/RLHF-loop workload, not single best-of-N.
    """
    print(f"\n[showcase] === SC8b: best-of-N COLD (unique prompt/call, no cache) ===", flush=True)
    max_new = 48
    # 5 unique ~0.9K prompts (best-of-5 cold single-call best-of-N).
    prompts = [build_unique_prompt(500 + r, target_words=650)
               + "\n\nWrite a concise technical summary.\n\nSummary:" for r in range(5)]
    # warmup the n=4 decode path with a throwaway unique prompt (no cache reuse).
    generate(build_unique_prompt(999, 650) + "\n\nSummary:",
             max_new=8, ignore_eos=True, n=4, temperature=0.7)
    best = 999
    for p in prompts:
        t0 = time.perf_counter()
        generate(p, max_new=max_new, ignore_eos=True, n=4, temperature=0.7)
        best = min(best, time.perf_counter() - t0)
    tps = (4 * max_new) / best
    print(f"[showcase] SC8b {engine_name}: best={best*1000:.0f}ms  tok/s={tps:.1f}  "
          f"n=4 x {max_new}tok (cold, unique prompt/call)", flush=True)
    return {"SC8b_cold_tps": f"{tps:.1f}"}


def run_sc10(engine_name, generate):
    """DL: SC10 — Shared system prompt, many tenants (aggregate throughput).

    12 independent short requests, all sharing the SAME ~0.9K-token
    system/context prefix (SHARED_PREFIX). The canonical production
    RadixAttention shape: one chatbot / agent system prompt served to many
    users. RadixAttention prefills the system prompt once and each tenant
    extends only its short question; vLLM (APC-off) re-prefills the system
    prompt on all 12. Reported as aggregate decode throughput (best-of-2 after
    warmup). Distinct from SC7 (~2K long-doc RAG) by a ~2x shorter prefix and
    more requests; from SC1 (per-request latency, 8 reqs) by throughput metric.
    """
    print(f"\n[showcase] === SC10: Shared System-Prompt Throughput (12 tenants) ===", flush=True)
    sys_prompt = SHARED_PREFIX
    questions = build_tenant_questions()[:12]   # 12 tenants share the system prompt
    total_decode = len(questions) * 24

    def one_pass():
        t0 = time.perf_counter()
        for q in questions:
            generate(sys_prompt + "\n\nQ: " + q + "\nA:", max_new=24, ignore_eos=True)
        return time.perf_counter() - t0

    one_pass()  # warmup: cache the shared system prompt once
    times = [one_pass() for _ in range(int(os.environ.get("SHOWCASE_REPS", "2")))]
    best = min(times)
    tps = total_decode / best
    print(f"[showcase] SC10 {engine_name}: tenants={len(questions)}  best={best*1000:.0f}ms  "
          f"throughput={tps:.1f} tok/s  reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)
    return {"SC10_throughput_tps": f"{tps:.1f}", "SC10_total_ms": f"{best*1000:.0f}"}


def run_sc11(engine_name, generate_batch):
    """DL: SC11 — Online concurrency (N tenants, shared system-prompt, UNEQUAL
    queries, concurrent batch). The production multi-tenant-concurrent shape that
    fix 0fe8c7cc86 (FA2 wrapper max_seqlen_q=max(extend_lens)) unblocked: N users
    behind one system prompt fire CONCURRENTLY (one batch), each a different-length
    query (unequal extend_lens -> the FA2 varlen path; pre-fix this crashed).
    Distinct from SC10 (same shape, SEQUENTIAL) by concurrency; from SC3 (equal-
    length batch) by unequal lengths. The harness caps concurrency at
    max_running_requests=4 (parity), so N tenants process in waves of 4. Aggregate
    decode throughput, best-of-N after warmup.
    """
    print(f"\n[showcase] === SC11: Online Concurrency (N tenants, shared sys-prompt, unequal queries) ===", flush=True)
    n = int(os.environ.get("SC11_USERS", "12"))
    sys_prompt = SHARED_PREFIX
    queries = build_sc11_queries()[:n]
    prompts = [sys_prompt + "\n\nQ: " + q + "\nA:" for q in queries]
    max_new = 24
    total_decode = len(prompts) * max_new

    def one_pass():
        t0 = time.perf_counter()
        generate_batch(prompts, max_new=max_new)  # concurrent batch -> N in-flight
        return time.perf_counter() - t0

    one_pass()  # warmup: cache shared sys-prompt + exercise the concurrent varlen path
    times = [one_pass() for _ in range(int(os.environ.get("SHOWCASE_REPS", "2")))]
    best = min(times)
    tps = total_decode / best
    print(f"[showcase] SC11 {engine_name}: tenants={len(prompts)} best={best*1000:.0f}ms  "
          f"throughput={tps:.1f} tok/s  reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)
    return {"SC11_throughput_tps": f"{tps:.1f}", "SC11_total_ms": f"{best*1000:.0f}"}


def run_sc9(engine_name, generate):
    """DL: SC9 — Pure long decode (short prompt) — decode-bound CONTROL / loss.

    A SHORT prompt (~14 tokens) + a 128-token single-stream greedy decode,
    best-of-3. Prefill is negligible, so this is dominated by raw decode TPOT —
    where vLLM holds its DLIN IPC edge. The honest counterpoint to SC5/7/8/10:
    with NO shared structure to reuse, vLLM's faster decode wins. Recorded as a
    known sglang loss (the decode-IPC gap). Expected: vLLM faster (loss).
    """
    print(f"\n[showcase] === SC9: Pure Long Decode (short prompt, 128 tok) [decode-bound] ===", flush=True)
    prompt = ("Write a detailed technical essay about the future of heterogeneous "
              "AI accelerators and their software stacks.")
    max_new = 128
    for _ in range(3):  # warmup the decode CG path
        generate(prompt, max_new=16, ignore_eos=True)
    best = 999
    for _ in range(3):
        t0 = time.perf_counter()
        generate(prompt, max_new=max_new, ignore_eos=True)
        best = min(best, time.perf_counter() - t0)
    tps = max_new / best
    print(f"[showcase] SC9 {engine_name}: best={best*1000:.0f}ms  tok/s={tps:.1f}  "
          f"{max_new}tok single-stream greedy", flush=True)
    return {"SC9_tps": f"{tps:.1f}"}


def run_sc6(engine_name, generate):
    """DL: SC6 — Raw-prefill parity (UNIQUE prompts, NO caching) — rigor probe.

    N unique ~1K-token prompts (distinct prefixes => RadixAttention CANNOT hit),
    each followed by a 4-token decode. Both engines prefill every prompt fully,
    so this isolates RAW prefill throughput. Purpose: determine whether the
    SC5/SC7/SC8/SC10 wins come from RadixAttention caching, or ALSO from sglang
    having faster raw-prefill kernels. If sglang ~= vLLM here, the prefix-scenario
    wins are PURELY caching (rigorous attribution). If sglang >> vLLM, raw prefill
    is an additional factor (and we check whether vLLM's slow prefill is a config
    artifact). Distinct prompts are used each pass so sglang gets no cross-pass
    cache hit. (Not a "showcase win" scenario — a diagnostic.)
    """
    print(f"\n[showcase] === SC6: Raw-Prefill Parity (8 unique ~1K prompts, no cache) ===", flush=True)
    # 3 passes (warmup + 2 measured) of 8 UNIQUE prompts each = 24 distinct prompts.
    pool = [build_unique_prompt(100 + p * 8 + k) for p in range(3) for k in range(8)]
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        tok_per = len(_tok.encode(pool[0]))
    except Exception:
        tok_per = int(len(pool[0].split()) * 1.4)
    total_prefill = 8 * tok_per

    def one_pass(offset):
        t0 = time.perf_counter()
        for k in range(8):
            generate(pool[offset * 8 + k], max_new=4, ignore_eos=True)
        return time.perf_counter() - t0

    one_pass(0)  # warmup (JIT)
    times = [one_pass(1), one_pass(2)]
    best = min(times)
    prefill_tps = total_prefill / best
    print(f"[showcase] SC6 {engine_name}: best={best*1000:.0f}ms  "
          f"raw_prefill~{prefill_tps:.0f} tok/s  ({tok_per}tok/prompt x8)  "
          f"reps={[f'{t*1000:.0f}' for t in times]} ms", flush=True)
    return {"SC6_prefill_tps": f"{prefill_tps:.0f}", "SC6_total_ms": f"{best*1000:.0f}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])
    parser.add_argument("--vllm-runner", default="mrv2", choices=["mrv1", "mrv2"],
                        help="vLLM model runner: mrv2 (V2, default) or mrv1 (V1). "
                             "MRV1 enforces eager to avoid the DLIN torch.compile crash.")
    parser.add_argument("--mem-frac", type=float, default=0.55)
    parser.add_argument("--scenarios", default="SC1,SC2,SC3",
                        help="comma list of SC1/SC2/SC3/SC4/SC5/SC7/SC8/SC10/SC11 "
                             "(default SC1,SC2,SC3). SC4=JSON. SC1 cold ~90s. "
                             "DL: SC5=multi-user fork (radix tree), "
                             "SC7=long-RAG throughput (3K prefix, 16 queries), "
                             "SC8=parallel sampling n=4 (decode-bound control), "
                             "SC10=shared system-prompt throughput (24 tenants), "
                             "SC11=online concurrency (N tenants, unequal queries, concurrent batch).")
    args = parser.parse_args()
    engine_name = args.engine
    runner = args.vllm_runner if engine_name == "vllm" else "sglang"
    enabled = {s.strip().upper() for s in args.scenarios.split(",") if s.strip()}
    unknown = enabled - {"SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC8B", "SC9", "SC10", "SC11"}
    if unknown:
        raise SystemExit(f"unknown scenario(s): {unknown} "
                         f"(valid: SC1 SC2 SC3 SC4 SC5 SC6 SC7 SC8 SC8B SC9 SC10 SC11)")

    print(f"[showcase] engine={engine_name} runner={runner} model={MODEL} tp={TP} "
          f"scenarios={sorted(enabled)}", flush=True)

    # ---- Launch engine ----
    raw = None  # the engine/llm object (sglang.Engine or vLLM LLM)
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
        raw = engine
        def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
            sp = {"max_new_tokens": max_new, "temperature": temperature, "ignore_eos": ignore_eos}
            if n > 1:
                sp["n"] = n
            return engine.generate(prompt, sp)
        def generate_batch(prompts, max_new=32, temperature=0.0, ignore_eos=True):
            return engine.generate(prompts, {"max_new_tokens": max_new,
                                             "temperature": temperature,
                                             "ignore_eos": ignore_eos})
        def shutdown():
            engine.shutdown()
    else:
        from vllm import LLM, SamplingParams
        # NOTE: enable_prefix_caching (APC) CANNOT be enabled on the compare's runner
        # (MRV2) for this hybrid Mamba model — vLLM forces mamba_cache_mode='align',
        # which MRV2 hard-rejects ("Model Runner V2 has not yet supported
        # mamba_cache_mode='align'", vllm/config/vllm.py:2030). MRV1 can enable APC
        # (eager) but is unstable on DLIN (this compare skips it). So vLLM runs
        # APC-OFF here (its only stable config) — re-prefilling shared prefixes.
        # Proven 2026-07-25 by scripts/dl/apc_failure_probe.py (MRV2+APC fails under
        # both CG and eager; MRV1+APC+eager works but MRV1 crashes on DLIN). Contrast:
        # sglang RadixAttention works natively + with CG. See fairness-defense doc.
        #
        # MRV1 vs MRV2: MRV2 (VLLM_USE_V2_MODEL_RUNNER=1 + CG) is the only
        # config that works on DLIN; MRV1 historically hits a torch.compile
        # dynamic-shape ConstraintViolationError, so we enforce_eager for MRV1
        # to give it a chance (still often fails — recorded as status=fail).
        mrv2 = (runner == "mrv2")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if mrv2 else "0"
        llm_kwargs = dict(
            model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
            max_model_len=4096, gpu_memory_utilization=args.mem_frac,
            trust_remote_code=True, max_num_seqs=4, disable_log_stats=True,
        )
        if mrv2:
            llm_kwargs["enforce_eager"] = False
            llm_kwargs["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4],
                                                "max_cudagraph_capture_size": 4}
        else:  # MRV1: dodge the torch.compile crash with eager
            llm_kwargs["enforce_eager"] = True
        llm = LLM(**llm_kwargs)
        raw = llm
        def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
            sp = SamplingParams(temperature=temperature, max_tokens=max_new,
                                ignore_eos=ignore_eos, n=n)
            out = llm.generate([prompt], sp)[0]
            if n > 1:
                return [o.text for o in out.outputs]
            return out.outputs[0].text
        def generate_batch(prompts, max_new=32, temperature=0.0, ignore_eos=True):
            sp = SamplingParams(temperature=temperature, max_tokens=max_new, ignore_eos=ignore_eos)
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
    metrics = {}

    # DL: run each scenario in a guard so one scenario's crash doesn't abort
    # the whole (long, JIT-heavy) run and wipe the other scenarios' metrics.
    # A failed scenario is logged and skipped (absent from the METRICS block).
    import traceback as _tb
    def _run(name, fn, *a, **kw):
        try:
            metrics.update(fn(*a, **kw))
        except Exception as e:  # noqa: BLE001 - benchmark must be resilient
            print(f"[showcase] !! {name} FAILED ({type(e).__name__}): {e}", flush=True)
            _tb.print_exc()
            # F4: persist an explicit per-scenario fail marker so a single-SC crash
            # (engine OK, one scenario threw) is recorded in the METRICS block +
            # JSON store + CSV, not just silently absent (both renderers already
            # surface a missing metric as FAIL/NA, but this disambiguates crash
            # from slow/zero).
            metrics[f"{name}_status"] = "fail"

    if "SC1" in enabled:
        _run("SC1", run_sc1, engine_name, generate, questions)
    if "SC2" in enabled:
        _run("SC2", run_sc2, engine_name, generate, turns)
    if "SC3" in enabled:
        _run("SC3", run_sc3, engine_name, generate_batch, questions)
    if "SC4" in enabled:
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
        _run("SC4", run_sc4, engine_name, raw, json_schema)
    if "SC5" in enabled:
        _run("SC5", run_sc5, engine_name, generate)
    if "SC6" in enabled:
        _run("SC6", run_sc6, engine_name, generate)
    if "SC7" in enabled:
        _run("SC7", run_sc7, engine_name, generate)
    if "SC8" in enabled:
        _run("SC8", run_sc8, engine_name, generate)
    if "SC8B" in enabled:
        _run("SC8B", run_sc8b, engine_name, generate)
    if "SC9" in enabled:
        _run("SC9", run_sc9, engine_name, generate)
    if "SC10" in enabled:
        _run("SC10", run_sc10, engine_name, generate)
    if "SC11" in enabled:
        _run("SC11", run_sc11, engine_name, generate_batch)

    # ---- Machine-readable metrics (parsed by run_sglang.sh `compare` phase) ----
    model_tag = os.path.basename(MODEL.rstrip("/"))
    print(f"\n=== METRICS engine={engine_name} runner={runner} model={model_tag} tp={TP} ===", flush=True)
    for k in sorted(metrics):
        print(f"METRIC {k}={metrics[k]}", flush=True)
    print("=== END METRICS ===", flush=True)

    print(f"\n[showcase] === {engine_name} SUMMARY ===", flush=True)
    for k, v in sorted(metrics.items()):
        print(f"  {k} = {v}", flush=True)

    shutdown()


if __name__ == "__main__":
    main()
