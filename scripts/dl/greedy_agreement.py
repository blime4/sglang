#!/usr/bin/env python3
"""greedy_agreement.py — do sglang and vLLM produce COMPARABLE greedy output?

`showcase_prefix_sharing.py` compares sglang vs vLLM on SPEED but never checks
that the two engines produce comparable OUTPUT on the same prompt at
temperature=0. They run different FP8 paths (sglang fused MoE vs vLLM `_dl_C`),
so greedy outputs *could* diverge — and if they do (different tokens / EOS
points / token counts), the speed comparison is apples-to-oranges. This probe
answers the foundational correctness question: is the comparison valid?

It also captures the natural (ignore_eos=False) completion length for the short
factual prompts — empirical evidence for the F1 fix (SC1/SC2/SC3 now force
ignore_eos=True): if the model emits EOS well before max_new, throughput/latency
computed against max_new is inflated, and if the two engines stop at different
points the comparison is unfair.

One engine per process (fresh, like `compare`):

  source sdk-dlop-07-13-20-30/env.sh
  SGLANG_DL_MOE_FUSED_MAX_M=2048 CUDA_VISIBLE_DEVICES=20,21,22,23 \\
      .venv/bin/python scripts/dl/greedy_agreement.py run --engine sglang --out /tmp/ga_sglang.json
  CUDA_VISIBLE_DEVICES=20,21,22,23 \\
      .venv/bin/python scripts/dl/greedy_agreement.py run --engine vllm --out /tmp/ga_vllm.json
  .venv/bin/python scripts/dl/greedy_agreement.py diff /tmp/ga_sglang.json /tmp/ga_vllm.json

Engine config below mirrors showcase main() / the run_sglang.sh compare preset
(qwen35-35b) — KEEP IN SYNC if that config changes (this is a diagnostic, so it
duplicates the config rather than refactor the production benchmark). Prompt set
is the real workload prompts imported from showcase (SC1/SC7/SC6/SC9 shapes).
"""
import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from showcase_prefix_sharing import (  # noqa: E402  — data only (prompts + MODEL/TP)
    SHARED_PREFIX, build_questions, build_long_doc, build_unique_prompt, MODEL, TP,
)

MAX_NEW = 48  # fixed decode length (ignore_eos) -> token counts comparable across engines


def launch_engine(engine_name, runner, mem_frac):
    """Launch sglang.Engine or vLLM LLM; return (generate, raw, shutdown).

    MIRRORS showcase main() config (TP/mem/CG/max_seqs/backend). Drift here would
    make the agreement check test a different config than the speed comparison.
    """
    if engine_name == "sglang":
        import sglang as sgl
        engine = sgl.Engine(
            model_path=MODEL, tp_size=TP, dtype="bfloat16",
            context_length=4096, mem_fraction_static=mem_frac,
            max_running_requests=4, disable_cuda_graph=False,
            cuda_graph_max_bs_decode=4, attention_backend="fa3", page_size=16,
            disable_custom_all_reduce=True, trust_remote_code=True,
            chunked_prefill_size=512,
        )
        def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
            sp = {"max_new_tokens": max_new, "temperature": temperature, "ignore_eos": ignore_eos}
            if n > 1:
                sp["n"] = n
            return engine.generate(prompt, sp)
        def shutdown():
            engine.shutdown()
        return generate, engine, shutdown

    from vllm import LLM, SamplingParams
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if runner == "mrv2" else "0"
    llm_kwargs = dict(
        model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=mem_frac,
        trust_remote_code=True, max_num_seqs=4, disable_log_stats=True,
        enforce_eager=runner != "mrv2",
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4], "max_cudagraph_capture_size": 4},
    )
    llm = LLM(**llm_kwargs)
    def generate(prompt, max_new=32, temperature=0.0, ignore_eos=False, n=1):
        sp = SamplingParams(temperature=temperature, max_tokens=max_new,
                            ignore_eos=ignore_eos, n=n)
        out = llm.generate([prompt], sp)[0]
        if n > 1:
            return [o.text for o in out.outputs]
        return out.outputs[0].text
    def shutdown():
        pass
    return generate, llm, shutdown


def _to_text(out):
    if isinstance(out, dict):
        t = out.get("text", str(out))
        return t[0] if isinstance(t, list) else t
    if isinstance(out, list):
        return out[0]
    return str(out)


def build_probe_prompts():
    """Fixed prompt set covering the showcase workload shapes.

    The first three are short factual Q&A over the shared prefix (SC1/SC3 shape)
    — also run naturally (ignore_eos=False) to measure early-EOS (F1 evidence).
    """
    qs = build_questions()
    doc = build_long_doc()
    return [
        ("sc1_short_a",  SHARED_PREFIX + "\n" + qs[0], True),
        ("sc1_short_b",  SHARED_PREFIX + "\n" + qs[1], True),
        ("sc1_short_c",  SHARED_PREFIX + "\n" + qs[2], True),
        ("sc7_long_doc", doc + "\n\nQ: What is the FP8 throughput per QUAD?\nA:", False),
        ("sc6_unique",   build_unique_prompt(100) + "\n\nWrite a concise technical summary.\n\nSummary:", False),
        ("sc9_short",    "Write a detailed technical essay about the future of heterogeneous "
                         "AI accelerators and their software stacks.", False),
    ]


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ga] tokenizer unavailable ({type(e).__name__}); ntok = word-count heuristic", flush=True)
        return None


def _ntok(tok, text):
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return len(text.split())


def cmd_run(args):
    tok = _load_tokenizer()
    generate, _raw, shutdown = launch_engine(args.engine, args.vllm_runner, args.mem_frac)
    print(f"[ga] {args.engine}/{args.vllm_runner if args.engine=='vllm' else 'sglang'} engine up; "
          f"FUSED_MAX_M={os.environ.get('SGLANG_DL_MOE_FUSED_MAX_M','?')} "
          f"max_new={MAX_NEW}; running probe prompts", flush=True)
    for _ in range(3):  # warmup (same as showcase)
        generate("Hello world", max_new=16)

    results = []
    for label, prompt, run_natural in build_probe_prompts():
        rec = {"label": label, "prompt_head": prompt[:80].replace("\n", " "),
               "prompt_len_chars": len(prompt)}
        t0 = time.perf_counter()
        rec["text"] = _to_text(generate(prompt, max_new=MAX_NEW, temperature=0.0, ignore_eos=True))
        rec["fixed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["fixed_ntok"] = MAX_NEW  # ignore_eos -> always MAX_NEW (sanity)
        if run_natural:
            t0 = time.perf_counter()
            nat = _to_text(generate(prompt, max_new=MAX_NEW, temperature=0.0, ignore_eos=False))
            rec["natural_text"] = nat
            rec["natural_ntok"] = _ntok(tok, nat)
            rec["natural_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(rec)
        natstr = f"  natural_ntok={rec.get('natural_ntok','-')}" if run_natural else ""
        print(f"[ga] {args.engine} {label}: fixed={rec['fixed_ms']}ms{natstr}  "
              f"out={rec['text'][:55]!r}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"engine": args.engine,
                   "runner": args.vllm_runner if args.engine == "vllm" else "sglang",
                   "model": os.path.basename(MODEL.rstrip("/")),
                   "fused_max_m": os.environ.get("SGLANG_DL_MOE_FUSED_MAX_M", "?"),
                   "max_new": MAX_NEW, "results": results}, f, indent=2)
    print(f"[ga] wrote {args.engine} -> {args.out}", flush=True)
    shutdown()


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def cmd_diff(args):
    a = json.load(open(args.a)); b = json.load(open(args.b))
    ra = {r["label"]: r for r in a["results"]}
    rb = {r["label"]: r for r in b["results"]}
    labels = [r["label"] for r in a["results"]]
    tok = _load_tokenizer()

    print(f"[ga-diff] {a['engine']}/{a.get('runner')} vs {b['engine']}/{b.get('runner')}  "
          f"(model {a.get('model')} / {b.get('model')}, max_new={a.get('max_new')}, "
          f"FUSED_MAX_M a={a.get('fused_max_m')} b={b.get('fused_max_m')})\n")
    hdr = f"  {'label':<14}{'exact':<7}{'tokLCP':<8}{'/48':<5}{'natA/natB':<11}{'verdict'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_exact = 0
    for label in labels:
        ta, tb = ra[label]["text"], rb.get(label, {}).get("text", "")
        exact = (ta == tb)
        if exact:
            n_exact += 1
        c = _lcp(ta, tb)
        tlcp = _ntok(tok, ta[:c]) if (tok and c) else 0  # token-LCP over common char-prefix
        nat_a = ra[label].get("natural_ntok")
        nat_b = rb.get(label, {}).get("natural_ntok")
        nat = f"{nat_a}/{nat_b}" if nat_a is not None else "-"
        if exact:
            verdict = "IDENTICAL"
        elif tlcp >= MAX_NEW * 0.8:
            verdict = "~equivalent (minor tail noise)"
        elif tlcp >= 5:
            verdict = f"diverge @~tok{tlcp}"
        else:
            verdict = "DIVERGE EARLY"
        print(f"  {label:<14}{'yes' if exact else 'no':<7}{tlcp:<8}{MAX_NEW:<5}{nat:<11}{verdict}")

    print(f"\n[ga-diff] exact match: {n_exact}/{len(labels)} prompts")
    print("\n  early-EOS (natural stop, max_new=48) — F1 evidence:")
    for label in labels:
        na, nb = ra[label].get("natural_ntok"), rb.get(label, {}).get("natural_ntok")
        if na is None:
            continue
        flag = "" if (na == nb) else "  <- engines STOP at DIFFERENT lengths (F2 fairness risk)"
        print(f"    {label:<14} sglang={na:<4} vllm={nb:<4}{flag}")
    print("\n  interpretation: identical/~equivalent -> comparison is sound; "
          "diverge-early -> investigate FP8 precision divergence.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one engine, write greedy outputs to JSON")
    r.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    r.add_argument("--vllm-runner", default="mrv2", choices=["mrv1", "mrv2"])
    r.add_argument("--mem-frac", type=float, default=0.55)
    r.add_argument("--out", required=True, help="output JSON path")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("diff", help="diff two engine JSONs (no GPU)")
    d.add_argument("a"); d.add_argument("b")
    d.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
