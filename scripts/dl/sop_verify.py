#!/usr/bin/env python3
"""SOP verification for DLIN (Denglin) sglang upgrades.

This is the judging standard ("评判标准") for porting DLIN changes onto a new
sglang upstream tag (dl-dev-v0.5.15 / dl-dev-v0.5.16). It runs a fixed set of
correctness + performance gates against the upgraded tree and emits a single
PASS/FAIL verdict + a JSON report. Exit code 0 = PASS, 2 = FAIL.

Why regression, not absolute: sglang greedy output diverges from vLLM on long
prefill (FP8 drift) and the DLIN stack has known-version-specific behaviors, so
"matches reference engine" is the wrong bar. An UPGRADE's job is to NOT REGRESS:
same model + same prompts on the new tag must (a) still produce coherent correct
answers on canonical short probes, and (b) stay within a noise band of the
known-good baseline's speed. So correctness is absolute (must pass) and perf is
relative (within tolerance of a recorded baseline).

Tiers:
  Correctness (absolute, ALL must pass):
    G1  DLIN stack smoke (torch.version.dl, GPU bf16 matmul, is_dlin, platform)
    G2  canonical probes (capital of France -> Paris, 1+1= -> 2, ...) greedy,
        loose-substring match (version-robust)
    G3  greedy determinism (same prompt twice -> byte-identical)
    G4  no-gibberish (degeneracy / repetition check on a long generation)
    G5  JSON structural correctness (generates + parses a JSON object)
  Regression (vs baseline golden, if recorded):
    R1  exact greedy output match of the canonical probes (catches subtle drift)
  Performance (vs baseline numbers, within tolerance band; skipped in --quick):
    P1  decode tok/s (amortized steady-state, median of 3)
    P2  prefill tok/s (1-token gen on a long prompt)

Modes (SOP_MODE / first CLI arg):
    verify  (default) run gates; gate perf vs baseline if present; PASS/FAIL
    record  run gates; SAVE correctness golden + perf numbers as the baseline
            (correctness must PASS for the baseline to be valid)
    show    re-print the latest report (no GPU run)

Assumes the clean DLIN runtime env is already active (run_sglang.sh sop sets it
via dlin_runtime_env): LD_LIBRARY_PATH=$SDK/lib ONLY, CUDA_HOME=$SDK, DLI_V2=ON.

Env (most set by run_sglang.sh::phase_sop):
    SOP_MODEL / SOP_TP / SOP_BACKEND / SOP_PAGE_SIZE / SOP_MEM_FRACTION
    SOP_CONTEXT_LEN / SOP_CG (1=on) / SOP_CG_MAX_BS / SOP_MAX_NEW
    SOP_MODE / SOP_BASELINE / SOP_QUICK / SOP_TOLERANCE / SOP_REPORT_DIR
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime

# Repo root (scripts/dl/ -> up two) so git lookups work regardless of cwd.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------- #
# Config (env-overridable; run_sglang.sh::phase_sop sets the DLIN-tuned values)
# --------------------------------------------------------------------------- #
MODEL = os.environ.get("SOP_MODEL", "/opt/dataset/Qwen3-1.7B")
TP = int(os.environ.get("SOP_TP", "1"))
BACKEND = os.environ.get("SOP_BACKEND", "fa3")
PAGE_SIZE = int(os.environ.get("SOP_PAGE_SIZE", "16"))
MEM_FRAC = float(os.environ.get("SOP_MEM_FRACTION", "0.80"))
CONTEXT_LEN = int(os.environ.get("SOP_CONTEXT_LEN", "4096"))
USE_CG = os.environ.get("SOP_CG", "0") == "1"
CG_MAX_BS = int(os.environ.get("SOP_CG_MAX_BS", "0"))
MAX_NEW = int(os.environ.get("SOP_MAX_NEW", "64"))
MODE = os.environ.get("SOP_MODE", "verify")
QUICK = os.environ.get("SOP_QUICK", "0") == "1"
# Perf regression tolerance: a gate PASSES if value >= baseline*(1-TOLERANCE).
# DLIN runs are noisy (SC9 +/-8 tok/s), so 0.20 decode / 0.30 prefill is the band.
TOL_DECODE = float(os.environ.get("SOP_TOL_DECODE", "0.20"))
TOL_PREFILL = float(os.environ.get("SOP_TOL_PREFILL", "0.30"))
REPORT_DIR = os.environ.get("SOP_REPORT_DIR", "/tmp/sglang_sop")

MODEL_TAG = os.path.basename(MODEL.rstrip("/"))

# Canonical short probes with version-robust loose-substring expectations.
# Qwen3 family (1.7B dense + 35B MoE) answer these identically when correct.
# "Answer:" priming cuts prompt-echo so the answer lands within max tokens.
PROBES = [
    {"tag": "fact_fr", "prompt": "The capital of France is",
     "expect_any": ["paris"], "max": 8},
    {"tag": "fact_cn", "prompt": "中国的首都是哪里？请用一个词回答。",
     "expect_any": ["北京", "beijing"], "max": 12},
    {"tag": "math_add", "prompt": "Question: What is 1 plus 1?\nAnswer:",
     "expect_any": ["2"], "max": 5},
    {"tag": "math_mul", "prompt": "Question: What is 15 times 4?\nAnswer:",
     "expect_any": ["60"], "max": 10},
    {"tag": "colors", "prompt": "List three primary colors.\nAnswer:",
     "expect_any": ["red", "blue", "yellow", "green"], "max": 24},
    {"tag": "english", "prompt": "Hello, how are you? Reply in one short sentence.\nAnswer:",
     "expect_any": [], "max": 40},  # no fixed expect; G4 gibberish check only
]
# Primed to emit JSON immediately: ends with the opening brace so the model
# completes the object (cuts instruction-ramble). gate_json prepends "{".
JSON_PROBE = 'Output ONLY a JSON object with keys "name" and "age" for Alice, age 30.\n{'
# A ~1K-token prompt for the prefill gate (repeated paragraph -> ~1K tokens).
PREFILL_PROMPT = ("The quick brown fox jumps over the lazy dog near the riverbank "
                  "where the tall green trees sway gently in the warm summer breeze. ") * 90


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Gate G1: DLIN stack smoke (absolute)
# --------------------------------------------------------------------------- #
def gate_smoke():
    try:
        import torch
        dl = getattr(torch.version, "dl", None)
        assert dl, f"torch.version.dl not set (torch={torch.__version__}); wrong wheel"
        assert torch.cuda.is_available(), "torch.cuda not available"
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        s = (a @ b).float().sum().item()
        torch.cuda.synchronize()
        import sglang
        from sglang.srt.platforms import current_platform
        from sglang.srt.utils.common import is_dlin
        assert is_dlin(), "is_dlin() is False"
        assert current_platform.is_dlin(), "current_platform.is_dlin() is False"
        plat = type(current_platform).__name__
        detail = (f"torch={torch.__version__} dl={dl} gpu={torch.cuda.get_device_name(0)} "
                  f"matmul_sum={s:.0f} platform={plat} sglang={sglang.__version__}")
        return {"gate": "G1_smoke", "status": "pass", "detail": detail,
                "torch_dl": dl, "platform": plat,
                "sglang_version": sglang.__version__}
    except Exception as e:  # noqa: BLE001
        return {"gate": "G1_smoke", "status": "fail",
                "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}


# --------------------------------------------------------------------------- #
# Engine construction (mirrors run_qwen3_1_7b.py / qwen35_sg_tps.py DLIN config)
# --------------------------------------------------------------------------- #
def build_engine():
    import sglang
    from sglang.srt.server_args import ServerArgs
    kwargs = dict(
        model_path=MODEL, page_size=PAGE_SIZE, dtype="bfloat16",
        attention_backend=BACKEND, disable_cuda_graph=not USE_CG,
        mem_fraction_static=MEM_FRAC, context_length=CONTEXT_LEN,
    )
    if TP > 1:
        # DL: TP>1 must use NCCL; the custom allreduce kernel hits HC_CUK Error=28.
        kwargs["tp_size"] = TP
        kwargs["disable_custom_all_reduce"] = True
    if USE_CG and CG_MAX_BS:
        kwargs["cuda_graph_max_bs_decode"] = CG_MAX_BS
    sa = ServerArgs(**kwargs)
    t0 = time.time()
    e = sglang.Engine(server_args=sa)
    log(f"[engine] {MODEL_TAG} loaded in {time.time()-t0:.1f}s "
        f"(tp={TP} backend={BACKEND} cg={USE_CG} mem={MEM_FRAC})")
    return e


def gen(engine, prompt, max_new, temperature=0):
    r = engine.generate(prompt, sampling_params={
        "max_new_tokens": max_new, "temperature": temperature})
    return r["text"] if isinstance(r, dict) else str(r)


_TOK = None


def _tokenizer():
    """Standalone tokenizer for prompt-length accounting (sglang.Engine's
    tokenizer accessor differs across versions; this is version-robust)."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


# --------------------------------------------------------------------------- #
# Correctness gates G2-G5 (absolute)
# --------------------------------------------------------------------------- #
def gate_probes(engine):
    """G2: canonical probes (loose substring). Returns (results, golden_outputs)."""
    results, golden = [], {}
    for p in PROBES:
        try:
            out = gen(engine, p["prompt"], p["max"])
            low = out.lower()
            golden[p["tag"]] = out
            if not p["expect_any"]:
                ok_g = True
            else:
                ok_g = any(x.lower() in low for x in p["expect_any"])
            gib = _gibberish_flag(out)
            status = "pass" if (ok_g and not gib) else "fail"
            results.append({
                "gate": f"G2_{p['tag']}", "status": status,
                "prompt": p["prompt"][:60], "got": out[:120].replace("\n", "\\n"),
                "expect_any": p["expect_any"], "gibberish": gib,
            })
        except Exception as e:  # noqa: BLE001
            results.append({"gate": f"G2_{p['tag']}", "status": "fail",
                            "detail": f"{type(e).__name__}: {e}"})
    return results, golden


def gate_determinism(engine):
    """G3: same prompt twice -> byte-identical (greedy)."""
    try:
        o1 = gen(engine, PROBES[0]["prompt"], 24)
        o2 = gen(engine, PROBES[0]["prompt"], 24)
        status = "pass" if o1 == o2 else "fail"
        return {"gate": "G3_determinism", "status": status,
                "identical": o1 == o2,
                "o1": o1[:60].replace("\n", "\\n"),
                "o2": o2[:60].replace("\n", "\\n")}
    except Exception as e:  # noqa: BLE001
        return {"gate": "G3_determinism", "status": "fail",
                "detail": f"{type(e).__name__}: {e}"}


def gate_gibberish(engine):
    """G4: degeneracy check on a long generation."""
    try:
        out = gen(engine, "Write a short paragraph (4 sentences) about the ocean.", 96)
        flag = _gibberish_flag(out)
        words = out.split()
        uniq_ratio = (len(set(words)) / len(words)) if words else 0.0
        status = "pass" if (not flag and uniq_ratio > 0.30) else "fail"
        return {"gate": "G4_no_gibberish", "status": status,
                "unique_word_ratio": round(uniq_ratio, 3),
                "got": out[:120].replace("\n", "\\n")}
    except Exception as e:  # noqa: BLE001
        return {"gate": "G4_no_gibberish", "status": "fail",
                "detail": f"{type(e).__name__}: {e}"}


def gate_json(engine):
    """G5: generates a parseable JSON object. The probe ends with '{', so prepend
    it back before parsing (the model completes the object after the brace)."""
    try:
        out = gen(engine, JSON_PROBE, 48)
        parsed, why = _try_parse_json("{" + out)
        status = "pass" if parsed is not None else "fail"
        return {"gate": "G5_json", "status": status,
                "parsed": parsed, "reason": why,
                "got": ("{" + out)[:120].replace("\n", "\\n")}
    except Exception as e:  # noqa: BLE001
        return {"gate": "G5_json", "status": "fail",
                "detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Performance gates P1/P2 (relative to baseline)
# --------------------------------------------------------------------------- #
def gate_decode_tps(engine):
    """P1: steady-state decode tok/s (warmup, then median of 3)."""
    try:
        gen(engine, "Hello", 32)  # warmup (amortize prefill JIT)
        tps = []
        for _ in range(3):
            t0 = time.time()
            out = gen(engine, "The future of AI is", MAX_NEW)
            dt = time.time() - t0
            tps.append(MAX_NEW / dt if dt > 0 else 0.0)
        med = statistics.median(tps)
        return {"gate": "P1_decode_tps", "status": "n/a", "value": round(med, 2),
                "unit": "tok/s", "samples": [round(x, 2) for x in tps]}
    except Exception as e:  # noqa: BLE001
        return {"gate": "P1_decode_tps", "status": "fail",
                "detail": f"{type(e).__name__}: {e}"}


def gate_prefill_tps(engine):
    """P2: raw prefill tok/s (1-token gen on a ~1K-token prompt)."""
    try:
        ntok = len(_tokenizer().encode(PREFILL_PROMPT))
        gen(engine, PREFILL_PROMPT, 1)  # warmup the prefill JIT for this shape
        t0 = time.time()
        gen(engine, PREFILL_PROMPT, 1)
        dt = time.time() - t0
        tps = ntok / dt if dt > 0 else 0.0
        return {"gate": "P2_prefill_tps", "status": "n/a", "value": round(tps, 2),
                "unit": "tok/s", "prompt_tokens": ntok}
    except Exception as e:  # noqa: BLE001
        return {"gate": "P2_prefill_tps", "status": "fail",
                "detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gibberish_flag(text):
    """Crude degeneracy detector: heavy token/n-gram repetition."""
    if not text.strip():
        return "empty"
    words = text.split()
    if len(words) >= 6:
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio < 0.30:
            return f"low_diversity({uniq_ratio:.2f})"
    # 8-char fragment repeated >3x in a row
    if re.search(r"(.{6,}?)\1{3,}", text):
        return "repetition_loop"
    return None


def _try_parse_json(text):
    """Parse JSON, tolerating leading/trailing prose by extracting the first {...}."""
    text = text.strip()
    try:
        return json.loads(text), "parsed"
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), "extracted"
        except Exception as e:
            return None, f"extract_failed: {e}"
    return None, "no_json_object"


def _git(field):
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--" + field, "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _meta():
    import torch
    import sglang
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git("short") if _git("short") != "unknown" else "unknown",
        "branch": _git("abbrev-ref"),
        "model": MODEL_TAG, "tp": TP, "backend": BACKEND,
        "cuda_graph": USE_CG, "page_size": PAGE_SIZE, "mem_fraction": MEM_FRAC,
        "sglang_version": sglang.__version__,
        "torch": torch.__version__,
        "torch_dl": getattr(torch.version, "dl", None),
    }


def _perf_band(result, baseline_val, tol, higher_better=True):
    """Mark a perf gate pass/fail vs baseline within tolerance. No baseline -> 'n/a'."""
    if baseline_val is None:
        return result
    val = result.get("value", 0.0)
    if higher_better:
        ratio = val / baseline_val if baseline_val else 0
        result["baseline"] = baseline_val
        result["ratio"] = round(ratio, 3)
        result["status"] = "pass" if val >= baseline_val * (1 - tol) else "fail"
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_all():
    smoke = gate_smoke()
    correctness = [smoke]
    perf, regression = [], []
    golden = {}
    if smoke["status"] == "pass":
        e = build_engine()
        try:
            res, golden = gate_probes(e)
            correctness.extend(res)
            correctness.append(gate_determinism(e))
            correctness.append(gate_gibberish(e))
            correctness.append(gate_json(e))
            if not QUICK:
                perf.append(gate_decode_tps(e))
                perf.append(gate_prefill_tps(e))
        finally:
            del e
    return correctness, perf, regression, golden


def load_baseline(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_report(report, baseline):
    m = report["meta"]
    print("\n" + "=" * 74)
    print(f"  SOP REPORT  {m['model']}  tp={m['tp']}  cg={m['cuda_graph']}  "
          f"{m['branch']}@{m['commit']}")
    print(f"  sglang={m['sglang_version']}  torch_dl={m['torch_dl']}  {m['timestamp']}")
    print("=" * 74)
    print(f"  {'gate':<22} {'status':<8} value")
    print("  " + "-" * 70)

    def row(g):
        val = ""
        if "value" in g:
            val = f"{g['value']} {g.get('unit','')}"
            if "baseline" in g:
                val += f"  (base {g['baseline']}, x{g.get('ratio','?')})"
        elif "got" in g:
            val = g["got"][:44]
        status = g["status"].upper()
        print(f"  {g['gate']:<22} {status:<8} {val}")
        if g["status"] == "fail" and "detail" in g:
            for line in g["detail"].splitlines()[:3]:
                print(f"  {'':22} {'':8}   {line}")

    for g in report["correctness"]:
        row(g)
    for g in report["regression"]:
        row(g)
    for g in report["perf"]:
        row(g)
    summ = report["summary"]
    print("  " + "-" * 70)
    print(f"  VERDICT: {report['verdict']}   "
          f"(passed {summ['passed']}/{summ['total']}, failed {summ['failed']})")
    if baseline is None and report["perf"]:
        print("  (perf shown but NOT gated: no baseline. Run 'sop record' on dl-main.)")
    print("=" * 74)


def main():
    global MODE, QUICK
    ap = argparse.ArgumentParser(description="DLIN sglang upgrade verification SOP")
    ap.add_argument("mode", nargs="?", default=MODE,
                    choices=["verify", "record", "show"])
    ap.add_argument("--baseline", default=os.environ.get("SOP_BASELINE", ""))
    ap.add_argument("--quick", action="store_true", default=QUICK)
    args = ap.parse_args()
    MODE = args.mode
    QUICK = args.quick or QUICK

    if MODE == "show":
        latest = os.path.join(REPORT_DIR, "latest.json")
        if not os.path.exists(latest):
            print("no report yet"); return 0
        with open(latest) as f:
            print_report(json.load(f), None)
        return 0

    os.makedirs(REPORT_DIR, exist_ok=True)
    # Default baseline path: one per model tag, under docs/dl/.
    if not args.baseline:
        args.baseline = f"docs/dl/sop_baseline_{MODEL_TAG}.json"
    baseline = load_baseline(args.baseline)

    correctness, perf, regression, golden = run_all()

    # --- Regression R1: exact greedy match vs golden (verify mode, baseline present)
    if MODE == "verify" and baseline and baseline.get("golden"):
        b_golden = baseline["golden"]
        mismatched = [t for t in b_golden if b_golden.get(t) != golden.get(t)]
        regression.append({
            "gate": "R1_exact_match", "status": "pass" if not mismatched else "fail",
            "mismatched": mismatched,
            "n_compared": len([t for t in b_golden if t in golden]),
        })

    # --- Perf band vs baseline (verify mode)
    if MODE == "verify" and baseline and baseline.get("perf"):
        bperf = {p["gate"]: p.get("value") for p in baseline["perf"]}
        for p in perf:
            tol = TOL_DECODE if "decode" in p["gate"] else TOL_PREFILL
            _perf_band(p, bperf.get(p["gate"]), tol)

    # --- Build report
    all_gates = correctness + regression + perf
    n_fail = sum(1 for g in all_gates if g["status"] == "fail")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    report = {
        "meta": _meta(), "correctness": correctness,
        "regression": regression, "perf": perf,
        "verdict": verdict,
        "summary": {"passed": len(all_gates) - n_fail, "failed": n_fail,
                    "total": len(all_gates)},
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"sop_{MODEL_TAG}_{stamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(REPORT_DIR, "latest.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- record mode: persist golden + perf as baseline (only if ALL gates OK)
    if MODE == "record":
        bad = sum(1 for g in (correctness + perf) if g["status"] == "fail")
        if bad:
            print(f"\n[record] REFUSED: {bad} gate(s) failed — "
                  "not a valid baseline. Fix and re-run.")
        else:
            base = {"meta": report["meta"], "golden": golden,
                    "perf": [{"gate": p["gate"], "value": p.get("value")}
                             for p in perf]}
            os.makedirs(os.path.dirname(args.baseline) or ".", exist_ok=True)
            with open(args.baseline, "w") as f:
                json.dump(base, f, indent=2, ensure_ascii=False)
            print(f"\n[record] baseline saved: {args.baseline}")
            baseline = base  # silence the "no baseline" note below

    print_report(report, baseline)
    print(f"\n[report] {report_path}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
