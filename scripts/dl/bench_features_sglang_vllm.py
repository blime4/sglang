#!/usr/bin/env python3
"""DL: feature-level sglang-vs-vLLM benchmark on DLIN (JSON structured output + prefix sharing).

Validates the two scenarios where the official SGLang blog claims an edge over vLLM:
  A. Structured output (JSON/regex) — Compressed FSM + xgrammar (sglang) vs structured_outputs (vLLM)
  B. Prefix sharing / multi-turn      — RadixAttention (sglang) vs prefix cache (vLLM)

Fairness contract (per .claude/skills/dl-compare-sglang-vllm §6⑦⑧):
  - SAME model + precision (FP8) on BOTH engines
  - SAME TP4, SAME GPUs, SEQUENTIAL (sglang in-process, then vLLM in a fresh subprocess)
  - best-of-N after warmup, temp=0

Scenario A isolates FSM overhead: measure tok/s for the SAME prompt constrained (JSON schema)
vs unconstrained. The constrained/unconstrained RATIO removes the baseline-decode-speed
difference (sglang decode is slower on DLIN), exposing pure FSM efficiency.

Scenario B isolates KV reuse: a long shared prefix + 6 short unique suffixes. Request 0 is cold
(cache empty), requests 1-5 are warm (cache hit). cold/warm wall-time ratio = cache benefit.

Run under sglang venv (vLLM is spawned as a subprocess):
    source sdk-dlop-07-13-20-30/env.sh
    CUDA_VISIBLE_DEVICES=20,21,22,23 .venv/bin/python scripts/dl/bench_features_sglang_vllm.py

Env overrides: SKIP_VLLM=1 / SKIP_SGLANG=1, MAX_TOKENS_JSON, TP (default 4), PORT (vLLM n/a here).
"""
import json
import os
import re
import subprocess
import sys
import time

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
# vLLM on DLIN is validated on the GPTQ-Int4 model (FP8 quant path hits
# "DL platform cant support double dtype(Empty op)"). sglang runs FP8; vLLM runs Int4.
# Different precision -> compare engine-internal FEATURE RATIOS (FSM overhead, cache
# speedup), not absolute tok/s. Override via env if a same-precision model is available.
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-GPTQ-Int4/")
TP = int(os.environ.get("TP", "4"))
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.60"))
MAX_JSON = int(os.environ.get("MAX_TOKENS_JSON", "96"))   # JSON decode length
N_PREFIX_OUT = int(os.environ.get("N_PREFIX_OUT", "16"))  # short output for prefix test
N_WARM = int(os.environ.get("N_WARM", "5"))               # warm prefix requests after the cold one
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", ".venv/bin/python")  # .venv has the DLIN-built vllm 0.21.1.dev2 + triton 3.3.0 natively
VLLM_OVERLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vllm-new-overlay")

# --- shared workload -------------------------------------------------------
JSON_PROMPT = (
    "You are a strict data extractor. Output ONLY a JSON object (no prose) describing the "
    "person in the text, matching the given schema exactly.\n\n"
    "Text: Dr. Ada Lovelace, 36, is a senior research scientist at DeepMind in London. "
    "She can be reached at ada.l@deepmind.example and her office is on the 5th floor.\n\n"
    "JSON object:"
)
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
        "employer": {"type": "string"},
        "city": {"type": "string"},
        "email": {"type": "string"},
    },
    "required": ["name", "age", "occupation", "employer", "city", "email"],
    "additionalProperties": False,
})

# A long shared prefix (~1-1.5K tokens): a system prompt + a passage + few-shot QA.
SHARED_PREFIX = (
    "You are a meticulous reading-comprehension assistant. Read the passage carefully and "
    "answer each subsequent question with a single short phrase.\n\n"
    "Passage: The Kahneman-Tversky collaboration, beginning in 1969, transformed behavioral "
    "economics by introducing prospect theory, which describes how people decide between "
    "alternatives that involve risk and uncertainty. Unlike expected utility theory, prospect "
    "theory accounts for the observed asymmetry between gains and losses: losses are felt "
    "roughly twice as intensely as equivalent gains, a phenomenon termed loss aversion. The "
    "fourfold pattern of risk attitudes predicts risk-aversion for high-probability gains and "
    "high-probability losses, alongside risk-seeking for low-probability gains and low-probability "
    "losses. Their work earned Kahneman the 2002 Nobel Memorial Prize in Economic Sciences; "
    "Tversky had died in 1996. The framing effect, another cornerstone of their research, shows "
    "that logically equivalent descriptions can produce systematically different choices, which "
    "has profound implications for public policy, medical decision-making, and negotiation. "
    "Anchoring, the heuristic whereby initial numerical values skew subsequent estimates, was "
    "demonstrated across domains from legal judgments to real-estate pricing. Availability, the "
    "tendency to judge frequency by the ease with which examples come to mind, explains why "
    "salient events are overestimated. Representativeness leads people to ignore base rates and "
    "sample size. Together these heuristics and biases constituted a descriptive alternative to "
    "the rational-agent model that dominated mid-twentieth-century economics. Subsequent work by "
    "Thaler, Ariely, and others extended these insights into nudge theory and choice architecture, "
    "influencing retirement savings plans, organ-donation defaults, and tax compliance. The dual-"
    "process framework, popularized in Kahneman's 2011 book Thinking, Fast and Slow, partitions "
    "cognition into a fast, intuitive System 1 and a slow, deliberate System 2, though the "
    "neuroscientific grounding of this dichotomy remains debated.\n\n"
    "Examples:\n"
    "Q: Which theory describes decisions under risk? A: prospect theory\n"
    "Q: How much more intensely are losses felt than gains? A: about twice\n"
    "Q: Who received the 2002 economics Nobel for this work? A: Daniel Kahneman\n"
    "Q: Which book popularized the dual-process framework? A: Thinking, Fast and Slow\n"
)
SUFFIX_QUESTIONS = [
    "Q: In what year did the collaboration begin? A:",
    "Q: What term describes the asymmetry between gains and losses? A:",
    "Q: Which heuristic ignores base rates and sample size? A:",
    "Q: Who extended these insights into nudge theory? A:",
    "Q: What does the framing effect imply about equivalent descriptions? A:",
    "Q: Which system in the dual-process framework is fast and intuitive? A:",
]


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# sglang backend (in-process Engine)
# ===========================================================================
def run_sglang():
    log("\n" + "=" * 70)
    log("  [sglang] FP8 TP4 — RadixAttention (on) + grammar_backend=xgrammar")
    log("=" * 70)

    # DLIN-proven env (compare_tp4.py / skill §3)
    for k, v in {
        "DLEOL_CACHE_SIZE": "1024",
        "DLEOL_FLA_ENABLE_PINGPONG": "1",
        "DLEOL_FLA_UNROLL_COUNT": "8",
        "SGLANG_DL_FP8_Q2": "1",
        "SGLANG_DL_MOE_FUSED": "1",
        "SGLANG_DL_MOE_FUSED_MAX_M": "32",
        "SGLANG_DL_GDN_DLIN": "1",
        "SGLANG_DL_MULTI_STEP": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }.items():
        os.environ.setdefault(k, v)

    import sglang
    from sglang.srt.server_args import ServerArgs

    _dl_kwargs = {}
    if os.environ.get("DL_CG_BACKEND_DECODE"):
        _dl_kwargs["cuda_graph_backend_decode"] = os.environ["DL_CG_BACKEND_DECODE"]
    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=MEM_FRAC,
        disable_cuda_graph=(os.environ.get("SGLANG_EAGER") == "1"),  # match vLLM eager when set
        cuda_graph_max_bs_decode=4,
        max_running_requests=4,
        context_length=4096,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        # grammar_backend defaults to xgrammar; radix cache ON by default
        **_dl_kwargs,
    )
    t0 = time.time()
    engine = sglang.Engine(server_args=sa)
    log(f"[sglang] loaded in {time.time()-t0:.1f}s")

    def gen(prompt, max_new, json_schema=None):
        sp = {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}
        if json_schema:
            sp["json_schema"] = json_schema
        t0 = time.perf_counter()
        out = engine.generate(prompt, sampling_params=sp)
        dt = time.perf_counter() - t0
        mi = out["meta_info"] if isinstance(out, dict) else {}
        n = mi.get("completion_tokens", max_new)
        return dt, n, out

    # ---- Scenario A: structured output (JSON) vs unconstrained ----
    # warmup both paths
    for _ in range(2):
        gen(JSON_PROMPT, 32)
        gen(JSON_PROMPT, 32, json_schema=JSON_SCHEMA)
    best_uc, best_c = 1e9, 1e9
    uc_tok = c_tok = 0
    uc_txt = c_txt = ""
    for _ in range(3):
        dt, n, out = gen(JSON_PROMPT, MAX_JSON)
        if dt < best_uc:
            best_uc, uc_tok, uc_txt = dt, n, out["text"] if isinstance(out, dict) else ""
        dt, n, out = gen(JSON_PROMPT, MAX_JSON, json_schema=JSON_SCHEMA)
        if dt < best_c:
            best_c, c_tok, c_txt = dt, n, out["text"] if isinstance(out, dict) else ""
    uc_tps = uc_tok / best_uc
    c_tps = c_tok / best_c
    log(f"[sglang-feature] scenario=A_unconstrained tok/s={uc_tps:.1f} n_tok={uc_tok} t={best_uc:.2f}s")
    log(f"[sglang-feature] scenario=A_json          tok/s={c_tps:.1f} n_tok={c_tok} t={best_c:.2f}s")
    log(f"[sglang-feature] scenario=A_fsm_overhead_ratio={c_tps/uc_tps:.3f} (constrained/unconstrained; 1.0=no overhead)")
    log(f"[sglang-feature] scenario=A_json_sample={c_txt[:160]!r}")

    # ---- Scenario B: prefix sharing (cold vs warm) ----
    # Warmup engine with an UNRELATED prompt so the shared prefix stays truly cold.
    import statistics
    UNRELATED = "Summarize the water cycle in three sentences."
    for _ in range(3):
        gen(UNRELATED, N_PREFIX_OUT)
    prompts = [SHARED_PREFIX + "\n" + q for q in SUFFIX_QUESTIONS[: 1 + N_WARM]]
    times = []
    for p in prompts:
        dt, n, _ = gen(p, N_PREFIX_OUT)
        times.append(dt)
    cold = times[0]                                  # p0: shared prefix not yet cached
    warm = statistics.median(times[1:])              # p1..pN: shared prefix cached -> robust to spikes
    log(f"[sglang-feature] scenario=B_cold  wall_s={cold:.3f} (first request, no cache)")
    log(f"[sglang-feature] scenario=B_warm  wall_s={warm:.3f} (median of {len(times)-1} cached)")
    log(f"[sglang-feature] scenario=B_cache_speedup={cold/warm:.2f}x (cold/warm; >1 = cache helps)")
    log(f"[sglang-feature] scenario=B_per_req={json.dumps([round(t,3) for t in times])}")

    engine.shutdown()
    return {
        "A_uc_tps": uc_tps, "A_c_tps": c_tps, "A_ratio": c_tps / uc_tps,
        "B_cold": cold, "B_warm": warm, "B_speedup": cold / warm,
        "A_json_sample": c_txt[:160],
    }


# ===========================================================================
# vLLM backend (fresh subprocess, MRV2)
# ===========================================================================
def run_vllm():
    log("\n" + "=" * 70)
    log("  [vLLM] FP8 TP4 — prefix cache (on) + structured_outputs (default backend)")
    log("=" * 70)

    script = f'''
import os, time, json
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("DLEOL_CACHE_SIZE", "1024")
os.environ.setdefault("DLEOL_FLA_ENABLE_PINGPONG", "1")
os.environ.setdefault("DLEOL_FLA_UNROLL_COUNT", "8")
os.environ.setdefault("DLEOL_USE_CU_MQA_TILEKV", "1")
os.environ.setdefault("VLLM_MAX_MOE_CU_TOKENS", "128")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
_vllm_eager = os.environ.get("VLLM_EAGER", "1") != "0"  # default eager; set VLLM_EAGER=0 to enable cuda-graph
_vllm_dtype = os.environ.get("VLLM_DTYPE", "half")  # FP8 model -> set VLLM_DTYPE=bfloat16
llm = LLM(model={VLLM_MODEL!r}, tensor_parallel_size={TP}, dtype=_vllm_dtype,
          max_model_len=4096, gpu_memory_utilization={MEM_FRAC}, trust_remote_code=True,
          enforce_eager=_vllm_eager, max_num_seqs=64, disable_log_stats=True,
          compilation_config={{"cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}})
JSON_PROMPT = {JSON_PROMPT!r}
JSON_SCHEMA = {JSON_SCHEMA!r}
MAX_JSON = {MAX_JSON}
SHARED_PREFIX = {SHARED_PREFIX!r}
SUFFIX = {SUFFIX_QUESTIONS!r}
N_PREFIX_OUT = {N_PREFIX_OUT}
N_WARM = {N_WARM}

def gen(prompt, max_tokens, structured=None):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    if structured is not None:
        sp.structured_outputs = structured
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)[0]
    dt = time.perf_counter() - t0
    n = len(out.outputs[0].token_ids)
    txt = out.outputs[0].text
    return dt, n, txt

# Scenario A
for _ in range(2):
    gen(JSON_PROMPT, 32)
    gen(JSON_PROMPT, 32, StructuredOutputsParams(json=JSON_SCHEMA))
best_uc, best_c = 1e9, 1e9
uc_tok = c_tok = 0
uc_txt = c_txt = ""
for _ in range(3):
    dt, n, txt = gen(JSON_PROMPT, MAX_JSON)
    if dt < best_uc: best_uc, uc_tok, uc_txt = dt, n, txt
    dt, n, txt = gen(JSON_PROMPT, MAX_JSON, StructuredOutputsParams(json=JSON_SCHEMA))
    if dt < best_c: best_c, c_tok, c_txt = dt, n, txt
uc_tps = uc_tok/best_uc; c_tps = c_tok/best_c
print(f"[vllm-feature] scenario=A_unconstrained tok/s={{uc_tps:.1f}} n_tok={{uc_tok}} t={{best_uc:.2f}}s", flush=True)
print(f"[vllm-feature] scenario=A_json          tok/s={{c_tps:.1f}} n_tok={{c_tok}} t={{best_c:.2f}}s", flush=True)
print(f"[vllm-feature] scenario=A_fsm_overhead_ratio={{c_tps/uc_tps:.3f}} (constrained/unconstrained; 1.0=no overhead)", flush=True)
print(f"[vllm-feature] scenario=A_json_sample={{c_txt[:160]!r}}", flush=True)

# Scenario B (noise-robust: unrelated warmup so p0 is truly cold; median for warm)
import statistics as _st
_UNRELATED = "Summarize the water cycle in three sentences."
for _ in range(3):
    gen(_UNRELATED, N_PREFIX_OUT)
prompts = [SHARED_PREFIX + "\\n" + q for q in SUFFIX[:1+N_WARM]]
times = []
for p in prompts:
    dt, n, txt = gen(p, N_PREFIX_OUT)
    times.append(dt)
cold = times[0]; warm = _st.median(times[1:])
print(f"[vllm-feature] scenario=B_cold  wall_s={{cold:.3f}} (first request, no cache)", flush=True)
print(f"[vllm-feature] scenario=B_warm  wall_s={{warm:.3f}} (median of {{len(times)-1}} cached)", flush=True)
print(f"[vllm-feature] scenario=B_cache_speedup={{cold/warm:.2f}}x (cold/warm; >1 = cache helps)", flush=True)
print(f"[vllm-feature] scenario=B_per_req={{json.dumps([round(t,3) for t in times])}}", flush=True)
print("[vllm-feature] DONE", flush=True)
'''
    env = os.environ.copy()
    if os.environ.get("USE_VLLM_OVERLAY") == "1" and os.path.isdir(VLLM_OVERLAY):
        env["PYTHONPATH"] = VLLM_OVERLAY + ":" + env.get("PYTHONPATH", "")
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # DL: run from a FILE, not `python -c`. vLLM V1 uses multiprocessing.spawn for
    # EngineCore/workers; spawn from a `-c` script fails to re-import the main module,
    # so workers start without the SDK env and can't find `dlcc` (JIT compiler) at
    # generation time. A real .py file lets spawn re-import correctly.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix="_vllm_bench.py", delete=False,
                                     dir="/tmp") as tf:
        tf.write(script)
        vllm_script_path = tf.name
    proc = subprocess.run([VLLM_PYTHON, vllm_script_path], env=env,
                          capture_output=True, text=True, timeout=2400)
    out_lines = (proc.stdout + proc.stderr).splitlines()
    for line in out_lines:
        if "[vllm-feature]" in line or "ERROR" in line.upper() or "Traceback" in line:
            log(line)
    if proc.returncode != 0:
        log(f"[vllm] FAILED rc={proc.returncode}; last 15 stderr lines:")
        for line in proc.stderr.splitlines()[-15:]:
            log(f"  {line}")
        return None
    # parse
    res = {}
    for line in proc.stdout.splitlines():
        m = re.search(r"scenario=A_unconstrained tok/s=([\d.]+)", line)
        if m: res["A_uc_tps"] = float(m.group(1))
        m = re.search(r"scenario=A_json          tok/s=([\d.]+)", line)
        if m: res["A_c_tps"] = float(m.group(1))
        m = re.search(r"scenario=B_cold  wall_s=([\d.]+)", line)
        if m: res["B_cold"] = float(m.group(1))
        m = re.search(r"scenario=B_warm  wall_s=([\d.]+)", line)
        if m: res["B_warm"] = float(m.group(1))
        m = re.search(r"scenario=A_json_sample=(.*)", line)
        if m: res["A_json_sample"] = m.group(1)
    if "A_uc_tps" in res and "A_c_tps" in res:
        res["A_ratio"] = res["A_c_tps"] / res["A_uc_tps"]
    if "B_cold" in res and "B_warm" in res:
        res["B_speedup"] = res["B_cold"] / res["B_warm"]
    return res or None


# ===========================================================================
def summarize(sg, vl):
    log("\n" + "#" * 70)
    log("  FEATURE-LEVEL COMPARISON  (sglang vs vLLM, same FP8 model, TP4, DLIN)")
    log("#" * 70)
    if not sg or not vl:
        log(f"  [incomplete] sglang={bool(sg)} vLLM={bool(vl)} — see logs above")
        return
    log("")
    log("  --- Scenario A: Structured output (JSON) — compressed FSM overhead ---")
    log(f"  sglang  unconstrained={sg['A_uc_tps']:.1f}  json={sg['A_c_tps']:.1f} tok/s   "
        f"FSM overhead ratio={sg['A_ratio']:.3f}")
    log(f"  vLLM    unconstrained={vl['A_uc_tps']:.1f}  json={vl['A_c_tps']:.1f} tok/s   "
        f"FSM overhead ratio={vl['A_ratio']:.3f}")
    log(f"  -> FSM efficiency winner (ratio closer to 1.0): "
        f"{'sglang' if sg['A_ratio']>vl['A_ratio'] else 'vLLM'}  "
        f"(sglang/vllm json-tok/s = {sg['A_c_tps']/vl['A_c_tps']:.2f}x)")
    log(f"  -> absolute json tok/s winner: "
        f"{'sglang' if sg['A_c_tps']>vl['A_c_tps'] else 'vLLM'}")
    log("")
    log("  --- Scenario B: Prefix sharing (cold vs warm KV reuse) ---")
    log(f"  sglang  cold={sg['B_cold']:.3f}s  warm={sg['B_warm']:.3f}s   "
        f"cache speedup={sg['B_speedup']:.2f}x")
    log(f"  vLLM    cold={vl['B_cold']:.3f}s  warm={vl['B_warm']:.3f}s   "
        f"cache speedup={vl['B_speedup']:.2f}x")
    log(f"  -> cache reuse winner (higher speedup): "
        f"{'sglang' if sg['B_speedup']>vl['B_speedup'] else 'vLLM'}")
    log(f"  -> absolute warm-request latency winner (lower=better): "
        f"{'sglang' if sg['B_warm']<vl['B_warm'] else 'vLLM'}")
    log("")
    log("  --- JSON sample check (both must be valid JSON of the schema) ---")
    log(f"  sglang: {sg.get('A_json_sample','')[:140]}")
    log(f"  vLLM  : {vl.get('A_json_sample','')[:140]}")


def main():
    log(f"Model: {MODEL}")
    log(f"TP={TP} GPUs={os.environ.get('CUDA_VISIBLE_DEVICES','all')} "
        f"MEM_FRAC={MEM_FRAC} MAX_JSON={MAX_JSON} N_PREFIX_OUT={N_PREFIX_OUT} N_WARM={N_WARM}")
    sg = vl = None
    if os.environ.get("SKIP_SGLANG") != "1":
        try:
            sg = run_sglang()
        except Exception as e:
            import traceback
            log(f"[sglang] EXCEPTION: {e}")
            traceback.print_exc()
    if os.environ.get("SKIP_VLLM") != "1":
        try:
            vl = run_vllm()
        except Exception as e:
            log(f"[vllm] EXCEPTION: {e}")
    summarize(sg, vl)
    # persist raw
    with open("bench_features_result.json", "w") as f:
        json.dump({"sglang": sg, "vllm": vl,
                   "config": {"model": MODEL, "tp": TP, "mem_frac": MEM_FRAC,
                              "max_json": MAX_JSON, "n_prefix_out": N_PREFIX_OUT,
                              "n_warm": N_WARM}}, f, indent=2)
    log("\n[persisted] bench_features_result.json")


if __name__ == "__main__":
    main()
