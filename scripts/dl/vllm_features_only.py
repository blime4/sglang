#!/usr/bin/env python3
# DL: standalone in-process vLLM feature benchmark (JSON structured output + prefix sharing).
# Runs as a TOP-LEVEL process so vLLM's EngineCore/worker multiprocessing.spawn inherits the
# SDK env. CRITICAL: all executable logic is under `if __name__ == '__main__':` — without it,
# spawn re-imports this module in each worker and re-runs LLM() (bootstrap RuntimeError).
#
# Usage:
#   source sdk-dlop-07-13-20-30/env.sh
#   CUDA_VISIBLE_DEVICES=28,29,30,31 .venv/bin/python scripts/dl/vllm_features_only.py
import os, time, json, statistics

MAX_JSON = int(os.environ.get("MAX_TOKENS_JSON", "96"))
N_PREFIX_OUT = int(os.environ.get("N_PREFIX_OUT", "16"))
N_WARM = int(os.environ.get("N_WARM", "5"))

JSON_PROMPT = ("You are a strict data extractor. Output ONLY a JSON object (no prose) describing the "
    "person in the text, matching the given schema exactly.\n\n"
    "Text: Dr. Ada Lovelace, 36, is a senior research scientist at DeepMind in London. "
    "She can be reached at ada.l@deepmind.example and her office is on the 5th floor.\n\nJSON object:")
JSON_SCHEMA = json.dumps({"type":"object","properties":{
    "name":{"type":"string"},"age":{"type":"integer"},"occupation":{"type":"string"},
    "employer":{"type":"string"},"city":{"type":"string"},"email":{"type":"string"}},
    "required":["name","age","occupation","employer","city","email"],"additionalProperties":False})
SHARED_PREFIX = ("You are a meticulous reading-comprehension assistant. Read the passage carefully and "
    "answer each subsequent question with a single short phrase.\n\n"
    "Passage: The Kahneman-Tversky collaboration, beginning in 1969, transformed behavioral economics "
    "by introducing prospect theory, which describes how people decide between alternatives that involve "
    "risk and uncertainty. Unlike expected utility theory, prospect theory accounts for the observed "
    "asymmetry between gains and losses: losses are felt roughly twice as intensely as equivalent gains, "
    "a phenomenon termed loss aversion. The framing effect shows that logically equivalent descriptions "
    "can produce systematically different choices. Anchoring skews estimates; availability overweights "
    "salient events; representativeness ignores base rates. Their work earned Kahneman the 2002 Nobel.\n\n"
    "Examples:\nQ: Which theory describes decisions under risk? A: prospect theory\n"
    "Q: How much more intensely are losses felt than gains? A: about twice\n"
    "Q: Who received the 2002 economics Nobel? A: Daniel Kahneman\n")
SUFFIX_Q = ["Q: In what year did the collaboration begin? A:",
            "Q: What term describes the gains/losses asymmetry? A:",
            "Q: Which heuristic ignores base rates? A:",
            "Q: What does the framing effect imply? A:",
            "Q: Which theory earned Kahneman the Nobel? A:",
            "Q: Name one heuristic that skews estimates. A:"]


def main():
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("DLEOL_CACHE_SIZE", "1024")
    os.environ.setdefault("DLEOL_FLA_ENABLE_PINGPONG", "1")
    os.environ.setdefault("DLEOL_FLA_UNROLL_COUNT", "8")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    MODEL = os.environ.get("VLLM_MODEL", "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/")
    TP = int(os.environ.get("TP", "4"))
    MEM = float(os.environ.get("MEM_FRAC", "0.6"))
    EAGER = os.environ.get("VLLM_EAGER", "1") != "0"
    DTYPE = os.environ.get("VLLM_DTYPE", "bfloat16")
    print(f"[vllm] model={MODEL} tp={TP} dtype={DTYPE} eager={EAGER} gpus={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype=DTYPE, max_model_len=4096,
              gpu_memory_utilization=MEM, trust_remote_code=True, enforce_eager=EAGER,
              max_num_seqs=64, disable_log_stats=True,
              compilation_config={"cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4})

    def gen(prompt, max_tokens, structured=None):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        if structured is not None:
            sp.structured_outputs = structured
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)[0]
        return time.perf_counter()-t0, len(out.outputs[0].token_ids), out.outputs[0].text

    # Scenario A: JSON structured output vs unconstrained
    for _ in range(2):
        gen(JSON_PROMPT, 32); gen(JSON_PROMPT, 32, StructuredOutputsParams(json=JSON_SCHEMA))
    best_uc = best_c = 1e9; uc_tok = c_tok = 0; c_txt = ""
    for _ in range(3):
        dt, n, _ = gen(JSON_PROMPT, MAX_JSON)
        if dt < best_uc: best_uc, uc_tok = dt, n
        dt, n, txt = gen(JSON_PROMPT, MAX_JSON, StructuredOutputsParams(json=JSON_SCHEMA))
        if dt < best_c: best_c, c_tok, c_txt = dt, n, txt
    uc_tps = uc_tok/best_uc; c_tps = c_tok/best_c
    print(f"[vllm-feature] scenario=A_unconstrained tok/s={uc_tps:.1f} n_tok={uc_tok} t={best_uc:.2f}s", flush=True)
    print(f"[vllm-feature] scenario=A_json          tok/s={c_tps:.1f} n_tok={c_tok} t={best_c:.2f}s", flush=True)
    print(f"[vllm-feature] scenario=A_fsm_overhead_ratio={c_tps/uc_tps:.3f}", flush=True)
    print(f"[vllm-feature] scenario=A_json_sample={c_txt[:160]!r}", flush=True)

    # Scenario B: prefix sharing (cold vs warm) — unrelated warmup so p0 is truly cold
    UNRELATED = "Summarize the water cycle in three sentences."
    for _ in range(3): gen(UNRELATED, N_PREFIX_OUT)
    prompts = [SHARED_PREFIX + "\n" + q for q in SUFFIX_Q[:1+N_WARM]]
    times = []
    for p in prompts:
        dt, n, txt = gen(p, N_PREFIX_OUT); times.append(dt)
    cold = times[0]; warm = statistics.median(times[1:])
    print(f"[vllm-feature] scenario=B_cold  wall_s={cold:.3f}", flush=True)
    print(f"[vllm-feature] scenario=B_warm  wall_s={warm:.3f} (median of {len(times)-1})", flush=True)
    print(f"[vllm-feature] scenario=B_cache_speedup={cold/warm:.2f}x", flush=True)
    print(f"[vllm-feature] scenario=B_per_req={json.dumps([round(t,3) for t in times])}", flush=True)
    print("[vllm-feature] DONE", flush=True)


if __name__ == "__main__":
    main()
