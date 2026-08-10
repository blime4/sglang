#!/usr/bin/env python3
"""DL: greedy (temperature=0) CORRECTNESS comparison of sglang vs vLLM on the
SAME model + SAME prompt(s). Answers "do the two engines produce bit-identical
greedy output for this model?" — not a perf benchmark.

Model-agnostic, but built for DeepSeek-V4-Flash on DLIN KS38. Each engine runs
in its OWN fresh process (invoked once per `run --engine`), mirroring the
compare phase's `_compare_run_one` pattern — required on DLIN for clean GPU /
cache state (a killed sglang/vLLM leaks VRAM and can corrupt the triton cache).

Flow (driven by run_sglang.sh `compare --correctness`):
  1. python compare_correctness.py run --engine sglang ... --out sglang.json
  2. python compare_correctness.py run --engine vllm   ... --out vllm.json
  3. python compare_correctness.py diff --sglang sglang.json --vllm vllm.json

Both sides use PLAIN greedy decode (no EAGLE/MTP). Speculative decoding is
lossless under greedy, but vLLM's MTP path crashes on DLIN under CG, so plain is
the stable, fair, apples-to-apples baseline. NOTE: the two engines use different
kernel implementations of the V4 ops (e.g. sglang's ported MHC Triton vs vLLM's
own) and may use different KV-cache precision (vLLM=fp8 here). Tiny FP8
reduction diffs can flip an argmax → divergence is a real, reportable finding,
not necessarily a bug. The diff step pinpoints the first diverging token.

Standalone (without run_sglang.sh):
  source <sdk-dlop>/env.sh && export CUDA_HOME=<sdk-dlop> CUDA_VISIBLE_DEVICES=0..7
  python scripts/dl/compare_correctness.py run --engine sglang \
      --model /LocalRun/hao.dong/DeepSeek-V4-Flash --tp 8 --out /tmp/c/sgl.json
  python scripts/dl/compare_correctness.py run --engine vllm \
      --model /LocalRun/hao.dong/DeepSeek-V4-Flash --tp 8 --out /tmp/c/vllm.json
  python scripts/dl/compare_correctness.py diff --sglang /tmp/c/sgl.json --vllm /tmp/c/vllm.json
"""
import argparse
import json
import os
import sys
import time
import traceback

# A small, diverse prompt set. Includes the vLLM V4 golden (prompt[1]) so the
# result can be cross-checked against vLLM's own correctness harness, and the
# canonical sglang V4 smoke prompt (prompt[0]). Used when --prompt/--prompts-file
# are not given.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "1 + 1 =",
    "用一句话解释相对论",
    "def fibonacci(n):\n    ",
]


def _to_id_list(v):
    """Coerce a tensor / list / tuple of token ids into a plain list[int]."""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return v


def _load_prompts(args):
    """Resolve the prompt list: --prompt (single) > --prompts-file > built-ins."""
    if getattr(args, "prompt", None):
        return [args.prompt]
    if getattr(args, "prompts_file", None):
        with open(args.prompts_file) as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]
        if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
            sys.exit(f"--prompts-file must be a JSON list of strings (got {type(data).__name__})")
        return data
    return list(DEFAULT_PROMPTS)


def _apply_v4_sglang_env():
    """Set the mandatory DeepSeek-V4 sglang env block (usage guide §8.2).

    These default the WRONG way in code (EnvBool True/False), so the whole block
    must be exported before constructing sgl.Engine on V4. setdefault so an
    explicit shell export always wins. Harmless on non-V4 models (the flags are
    V4-specific code paths that other models never enter).
    """
    block = {
        "SGLANG_DL_MOE_FUSED": "1",
        "SGLANG_DL_FP8_Q2": "1",
        "SGLANG_DL_GDN_DLIN": "1",
        "SGLANG_DL_MOE_FUSED_MAX_M": "2048",
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
        "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
        # These default True and route to unported DLIN paths -> must be 0.
        "SGLANG_OPT_USE_TOPK_V2": "0",
        "SGLANG_OPT_USE_FUSED_HASH_TOPK": "0",
        "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "0",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
        "SGLANG_DL_IDX_TRITON": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }
    for k, v in block.items():
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# sglang side
# ---------------------------------------------------------------------------
def run_sglang(args):
    _apply_v4_sglang_env()
    import sglang as sgl

    print(f"[sglang] model={args.model} tp={args.tp} ctx={args.max_model_len} "
          f"max_tokens={args.max_tokens} ignore_eos={args.ignore_eos} @ {time.strftime('%H:%M:%S')}",
          flush=True)
    # Plain greedy V4 recipe (usage guide §5.3): disable-prefill-CG keeps decode
    # CG; NO speculative -> clean comparison vs vLLM plain. trust_remote_code +
    # disable_custom_all_reduce (NCCL) match the proven serving config.
    engine_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("SGLANG_MEM_FRAC", "0.90")),
        context_length=args.max_model_len,
        disable_prefill_cuda_graph=True,
        cuda_graph_max_bs_decode=1,
        disable_custom_all_reduce=True,
    )
    # Match vLLM's KV-cache precision (vLLM uses fp8). Default "auto" = unset.
    if getattr(args, "kv_cache_dtype", "auto") != "auto":
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    t0 = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    print(f"[sglang] engine up in {time.perf_counter()-t0:.0f}s", flush=True)

    tok = engine.tokenizer_manager.tokenizer if hasattr(engine, "tokenizer_manager") else None
    prompts = _load_prompts(args)

    def gen(prompt):
        sp = {"max_new_tokens": args.max_tokens, "temperature": 0}
        if args.ignore_eos:
            sp["ignore_eos"] = True
        return engine.generate(prompt, sampling_params=sp)

    # one short warmup (first call JIT-compiles on DLIN)
    print("[sglang] warmup ...", flush=True)
    try:
        _ = gen(prompts[0])
    except Exception as e:
        print(f"[sglang] warmup failed (continuing): {type(e).__name__}: {e}", flush=True)

    results = []
    for i, p in enumerate(prompts):
        out = gen(p)
        text = getattr(out, "text", None) or (out["text"] if isinstance(out, dict) else str(out))
        # Probe the usual keys for the raw generated token ids (engines differ).
        ids = None
        obj = out if not isinstance(out, dict) else out
        for attr in ("output_ids", "output_token_ids", "token_ids"):
            v = getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)
            if v is not None:
                ids = _to_id_list(v)
                break
        # Fallback: derive ids from the text so diff still has something to line
        # up if the engine didn't surface raw ids.
        if ids is None and tok is not None:
            try:
                enc = tok.encode(text) if isinstance(text, str) else None
                ids = _to_id_list(enc["input_ids"] if isinstance(enc, dict) else enc)
            except Exception:
                ids = None
        print(f"[sglang] prompt[{i}] -> {str(text)[:80]!r} ({len(ids) if ids else '?'} ids)", flush=True)
        results.append({"prompt": p, "text": text, "token_ids": ids})

    engine.shutdown()
    _dump(args.out, engine="sglang", args=args, results=results)
    print(f"[sglang] SAVED {args.out}", flush=True)


# ---------------------------------------------------------------------------
# vLLM side
# ---------------------------------------------------------------------------
def run_vllm(args):
    # vLLM's multiprocessing.spawn re-imports __main__; the LLM build + generate
    # must live under the __main__ guard (caller enforces this, but keep the
    # imports local too so a stray import elsewhere can't trigger spawn issues).
    from vllm import LLM, SamplingParams

    print(f"[vllm] model={args.model} tp={args.tp} max_model_len={args.max_model_len} "
          f"max_tokens={args.max_tokens} ignore_eos={args.ignore_eos} @ {time.strftime('%H:%M:%S')}",
          flush=True)
    # Plain (non-spec) V4 config (scripts/dl/v4_vllm_bench.py): the only stable
    # vLLM V4 config on DLIN — MTP crashes under CG. kv_cache_dtype=fp8 matches
    # the proven recipe. enforce_eager off -> CG on (deterministic, faster).
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=float(os.environ.get("VLLM_MEM_FRAC", "0.90")),
        kv_cache_dtype="fp8",
        enforce_eager=(os.environ.get("VLLM_EAGER", "0") == "1"),
        disable_custom_all_reduce=True,
    )
    print("[vllm] engine up", flush=True)

    prompts = _load_prompts(args)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    if args.ignore_eos:
        sp.ignore_eos = True

    # warmup
    print("[vllm] warmup ...", flush=True)
    try:
        llm.generate([prompts[0]], SamplingParams(max_tokens=8, temperature=0))
    except Exception as e:
        print(f"[vllm] warmup failed (continuing): {type(e).__name__}: {e}", flush=True)

    outs = llm.generate(prompts, sp)
    results = []
    for i, out in enumerate(outs):
        comp = out.outputs[0]
        text = comp.text
        ids = _to_id_list(comp.token_ids)
        print(f"[vllm] prompt[{i}] -> {str(text)[:80]!r} ({len(ids) if ids else '?'} ids)", flush=True)
        results.append({"prompt": prompts[i], "text": text, "token_ids": ids})

    _dump(args.out, engine="vllm", args=args, results=results)
    print(f"[vllm] SAVED {args.out}", flush=True)


def _dump(path, engine, args, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "engine": engine,
        "model": args.model,
        "tp": args.tp,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "ignore_eos": args.ignore_eos,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def _first_text_diff(a, b):
    """Index of the first differing char in strings a,b (or min len if one is a prefix)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def diff(args):
    with open(args.sglang) as f:
        sgl = json.load(f)
    with open(args.vllm) as f:
        vll = json.load(f)
    rs, rv = sgl["results"], vll["results"]

    print()
    print("=" * 78)
    print("  GREEDY (temperature=0) CORRECTNESS: sglang vs vLLM")
    print("=" * 78)
    print(f"  model       : {sgl.get('model')}")
    print(f"  tp          : sglang={sgl.get('tp')}  vllm={vll.get('tp')}")
    print(f"  max_tokens  : {sgl.get('max_tokens')}   ignore_eos: {sgl.get('ignore_eos')}")
    print(f"  prompts     : {len(rs)} (sglang) vs {len(rv)} (vllm)")
    print("-" * 78)

    n_cmp = min(len(rs), len(rv))
    n_match_text = 0
    n_match_ids = 0
    for i in range(n_cmp):
        a, b = rs[i], rv[i]
        ta, tb = a.get("text") or "", b.get("text") or ""
        ia, ib = a.get("token_ids"), b.get("token_ids")
        same_text = ta == tb
        same_ids = (ia is not None and ib is not None and ia == ib)
        if same_text:
            n_match_text += 1
        if same_ids:
            n_match_ids += 1

        tag = "IDENTICAL" if same_text else "DIVERGE"
        print(f"  [{tag}] prompt {i}: {a.get('prompt','')[:60]!r}")
        print(f"      sglang: {ta[:90]!r}")
        print(f"      vllm  : {tb[:90]!r}")
        if not same_text:
            di = _first_text_diff(ta, tb)
            print(f"      first char diff @ {di}: sglang {ta[di:di+12]!r} vs vllm {tb[di:di+12]!r}")
        if ia is not None and ib is not None:
            if ia == ib:
                print(f"      token_ids: identical ({len(ia)} tokens)")
            else:
                # first diverging token id
                m = min(len(ia), len(ib))
                k = next((j for j in range(m) if ia[j] != ib[j]), m if len(ia) != len(ib) else -1)
                print(f"      token_ids DIVERGE @ idx {k}: sglang={ia[k] if k>=0 else None} "
                      f"vllm={ib[k] if k>=0 else None}  (len {len(ia)} vs {len(ib)})")
        else:
            print("      token_ids: <unavailable on one side; text comparison above is authoritative>")
    if len(rs) != len(rv):
        print(f"  NOTE: prompt-count mismatch (sglang {len(rs)} vs vllm {len(rv)}); "
              f"compared first {n_cmp}.")

    print("-" * 78)
    verdict_text = "PASS" if n_match_text == n_cmp and len(rs) == len(rv) else "DIFFER"
    print(f"  TEXT  identical: {n_match_text}/{n_cmp}   -> {verdict_text}")
    if n_match_ids or any(r.get("token_ids") is not None for r in rs + rv):
        verdict_ids = "PASS" if n_match_ids == n_cmp and len(rs) == len(rv) else "DIFFER"
        print(f"  TOKEN identical: {n_match_ids}/{n_cmp}   -> {verdict_ids}")
    print("=" * 78)
    if args.verbose:
        print(json.dumps({"sglang": sgl, "vllm": vll}, indent=2, default=str))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # `run --engine` (single parser; cleaner than two near-identical subparsers)
    pr = sub.add_parser("run", help="generate from one engine, dump JSON")
    pr.add_argument("--engine", required=True, choices=["sglang", "vllm"])
    pr.add_argument("--model", required=True)
    pr.add_argument("--tp", type=int, default=8)
    pr.add_argument("--max-tokens", type=int, default=64)
    pr.add_argument("--max-model-len", type=int, default=2048)
    pr.add_argument("--kv-cache-dtype", default="auto",
                    help="sglang kv_cache_dtype (default auto=unset; 'fp8' to match vLLM)")
    pr.add_argument("--prompt", default=None, help="single prompt (default: built-in set)")
    pr.add_argument("--prompts-file", default=None, help="JSON list of prompts")
    pr.add_argument("--ignore-eos", action="store_true", help="generate exactly max_tokens (fixed length)")
    pr.add_argument("--out", required=True)

    pd = sub.add_parser("diff", help="compare two run outputs")
    pd.add_argument("--sglang", required=True)
    pd.add_argument("--vllm", required=True)
    pd.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    try:
        if args.cmd == "run":
            if args.engine == "sglang":
                run_sglang(args)
            else:
                run_vllm(args)
        elif args.cmd == "diff":
            diff(args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[{getattr(args,'engine',args.cmd)}] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        out = getattr(args, "out", None)
        if out:
            with open(out + ".fail", "w") as f:
                f.write(f"{type(e).__name__}: {e}\n\n")
                traceback.print_exc(file=f)
        sys.exit(2)


if __name__ == "__main__":
    main()
