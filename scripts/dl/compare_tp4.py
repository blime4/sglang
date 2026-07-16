#!/usr/bin/env python3
"""TP4 sglang vs vLLM TPOT comparison (bug 18025 style).

Runs both engines offline on the same prompt/config, reports TTFT + TPOT side by
side. Designed for Qwen3.5-35B-A3B-FP8 on DLIN KS38.

Usage:
    source $SDK_DIR/env.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/dl/compare_tp4.py

Env overrides:
    PROMPT          — input text (default: short factual)
    MAX_NEW_TOKENS  — decode length (default: 128)
    MEM_FRAC        — sglang mem_fraction_static (default: 0.60)
    MEM_UTIL        — vLLM gpu_memory_utilization (default: 0.6)
    SKIP_VLLM=1    — skip vLLM run (debug sglang only)
    SKIP_SGLANG=1  — skip sglang run (debug vLLM only)
    VLLM_PYTHON     — path to vLLM venv python (auto-detected)
"""
import os
import subprocess
import sys
import time

MODEL = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
PROMPT = os.environ.get("PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TP = 4
MEM_FRAC = float(os.environ.get("MEM_FRAC", "0.60"))
MEM_UTIL = float(os.environ.get("MEM_UTIL", "0.6"))
VLLM_PYTHON = os.environ.get(
    "VLLM_PYTHON",
    "/LocalRun/ming.duan/llm/llm_20260629/.venv/bin/python",
)


def log(msg):
    print(msg, flush=True)


def run_sglang():
    """Run sglang TP4 offline benchmark via Engine API."""
    log("\n" + "=" * 60)
    log("  sglang TP4 (CG + invoke_fused_moe_opt + vllm_flash_attn)")
    log("=" * 60)

    os.environ.setdefault("DLEOL_CACHE_SIZE", "1024")
    os.environ.setdefault("DLEOL_CU_ADDRESS_CHECK", "0")
    os.environ.setdefault("SGLANG_DL_FP8_Q2", "1")
    os.environ.setdefault("SGLANG_DL_MOE_FUSED", "1")
    os.environ.setdefault("SGLANG_DL_MOE_FUSED_MAX_M", "16")

    import sglang
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(
        model_path=MODEL,
        dtype="bfloat16",
        tp_size=TP,
        attention_backend="fa3",
        page_size=16,
        mem_fraction_static=MEM_FRAC,
        disable_cuda_graph=False,
        cuda_graph_max_bs_decode=2,
        context_length=4096,
        disable_custom_all_reduce=True,
    )
    t_load = time.time()
    engine = sglang.Engine(server_args=sa)
    t_load = time.time() - t_load
    log(f"[sglang] loaded in {t_load:.1f}s")

    # Warmup (JIT + graph capture already done, but warm the decode path)
    for _ in range(3):
        engine.generate(PROMPT, sampling_params={"max_new_tokens": 8, "temperature": 0})

    # Streaming measurement
    sp = {"max_new_tokens": MAX_NEW, "temperature": 0}
    t0 = time.time()
    stream = engine.generate(PROMPT, sampling_params=sp, stream=True)
    ttft = None
    itls = []
    last = None
    text_out = ""
    for chunk in stream:
        text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        if text:
            now = time.time()
            text_out += text
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - last)
            last = now
    total = time.time() - t0
    engine.shutdown()

    import statistics
    med_itl = statistics.median(itls) * 1000 if itls else 0
    mean_itl = statistics.mean(itls) * 1000 if itls else 0
    n_tok = len(itls) + 1
    wall_tpot = total / n_tok * 1000
    decode_tpot = (total - ttft) / max(n_tok - 1, 1) * 1000

    log(f"[sglang] TTFT={ttft*1000:.1f}ms | decode_TPOT={decode_tpot:.1f}ms | "
        f"wall_TPOT={wall_tpot:.1f}ms | {n_tok/total:.1f} tok/s")
    log(f"[sglang] ITL median={med_itl:.1f}ms mean={mean_itl:.1f}ms over {len(itls)} chunks")
    log(f"[sglang OUT] {text_out[:100]!r}")
    return {"ttft_ms": ttft * 1000, "decode_tpot_ms": decode_tpot,
            "wall_tpot_ms": wall_tpot, "toks": n_tok / total}


def run_vllm():
    """Run vLLM TP4 offline benchmark in a subprocess (separate venv)."""
    log("\n" + "=" * 60)
    log("  vLLM TP4 (CG + torch.compile + inductor fusions)")
    log("=" * 60)

    script = f'''
import os, time
os.environ["DLEOL_CACHE_SIZE"] = "1024"
from vllm import LLM, SamplingParams
llm = LLM(model="{MODEL}", tensor_parallel_size={TP}, dtype="bfloat16",
          max_model_len=4096, gpu_memory_utilization={MEM_UTIL},
          trust_remote_code=True, disable_log_stats=True)
prompt = """{PROMPT}"""
sp = SamplingParams(temperature=0, max_tokens={MAX_NEW})
# Warmup x3
for _ in range(3):
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8))
# Timed
t0 = time.time()
out = llm.generate([prompt], sp)[0]
dt = time.time() - t0
n = len(out.outputs[0].token_ids)
wall_tpot = dt / n * 1000
print(f"[vllm] wall_TPOT={{wall_tpot:.1f}}ms | {{n/dt:.1f}} tok/s | tokens={{n}} total={{dt:.3f}}s")
print(f"[vllm OUT] {{out.outputs[0].text[:100]!r}}")
'''
    env = os.environ.copy()
    result = subprocess.run(
        [VLLM_PYTHON, "-c", script],
        env=env, capture_output=True, text=True, timeout=600,
    )
    # Extract results from stdout
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("[vllm]") or line.startswith("[vllm OUT]"):
            log(line)
    if result.returncode != 0:
        log(f"[vllm] ERROR (rc={result.returncode})")
        for line in result.stderr.splitlines()[-5:]:
            log(f"  {line}")
        return None

    # Parse
    for line in result.stdout.splitlines():
        if "[vllm] wall_TPOT=" in line:
            import re
            m = re.search(r"wall_TPOT=([\d.]+)ms.*?([\d.]+) tok/s", line)
            if m:
                return {"wall_tpot_ms": float(m.group(1)), "toks": float(m.group(2))}
    return None


def main():
    log(f"Model: {MODEL}")
    log(f"TP={TP} | prompt={PROMPT!r} | max_new_tokens={MAX_NEW}")
    log(f"GPUs: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    sg = None
    vl = None

    if os.environ.get("SKIP_SGLANG") != "1":
        sg = run_sglang()
    if os.environ.get("SKIP_VLLM") != "1":
        vl = run_vllm()

    log("\n" + "=" * 60)
    log("  COMPARISON SUMMARY")
    log("=" * 60)
    if sg:
        log(f"  sglang TP4: decode_TPOT={sg['decode_tpot_ms']:.1f}ms | "
            f"wall_TPOT={sg['wall_tpot_ms']:.1f}ms | {sg['toks']:.1f} tok/s")
    if vl:
        log(f"  vLLM   TP4: wall_TPOT={vl['wall_tpot_ms']:.1f}ms | {vl['toks']:.1f} tok/s")
    if sg and vl:
        gap = sg["wall_tpot_ms"] - vl["wall_tpot_ms"]
        ratio = sg["wall_tpot_ms"] / vl["wall_tpot_ms"]
        log(f"  Gap: {gap:.1f}ms (sglang {ratio:.2f}x slower)")
    log("=" * 60)


if __name__ == "__main__":
    main()
