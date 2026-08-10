#!/usr/bin/env python
# DLIN DFLASH correctness + perf probe for Qwen3.5-35B-A3B-FP8.
# Usage:
#   CUDA_VISIBLE_DEVICES=16,17,18,19 python scripts/dl/dflash_correctness.py plain
#   CUDA_VISIBLE_DEVICES=16,17,18,19 python scripts/dl/dflash_correctness.py dflash
# Each mode writes /tmp/dflash-repro/<mode>.json .
import os
import sys
import json
import time
import traceback

TARGET = "/mars/aebox/LLM/model/Qwen3.5-35B-A3B-FP8/"
DRAFT = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang/models-dl/Qwen3.5-35B-A3B-DFlash"
PROMPT = os.environ.get("DFLASH_PROMPT", "The quick brown fox jumps over the lazy dog.")
MAX_NEW = int(os.environ.get("DFLASH_MAX_NEW", "64"))


def _to_id_list(v):
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return v


def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
    assert MODE in ("plain", "dflash"), f"unknown mode {MODE}"
    OUT = f"/tmp/dflash-repro/{MODE}.json"
    os.makedirs("/tmp/dflash-repro", exist_ok=True)

    print(f"[{MODE}] target={TARGET}", flush=True)
    if MODE == "dflash":
        print(f"[{MODE}] draft ={DRAFT}", flush=True)
    print(f"[{MODE}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    try:
        import sglang as sgl

        common = dict(
            model_path=TARGET,
            tp_size=4,
            dtype="bfloat16",
            attention_backend="fa3",
            page_size=16,
            context_length=4096,          # DLIN recipe: bound KV memory (not 262144)
            mem_fraction_static=0.60,     # DLIN recipe: 0.60, NOT 0.70 (OOM on 32GB cards)
            disable_cuda_graph=False,     # CG on (matches parity recipe)
            disable_custom_all_reduce=True,
            trust_remote_code=True,
        )
        if MODE == "dflash":
            common.update(
                speculative_algorithm="DFLASH",
                speculative_draft_model_path=DRAFT,
                speculative_dflash_block_size=16,
            )

        t0 = time.time()
        engine = sgl.Engine(**common)
        print(f"[{MODE}] engine built in {time.time()-t0:.1f}s", flush=True)

        tok = engine.tokenizer_manager.tokenizer if hasattr(engine, "tokenizer_manager") else None

        def gen(max_new):
            return engine.generate(
                PROMPT, sampling_params={"max_new_tokens": max_new, "temperature": 0}
            )

        print(f"[{MODE}] warmup ...", flush=True)
        _ = gen(8)

        # ---- correctness run ----
        out = gen(MAX_NEW)
        text = getattr(out, "text", None) or (out["text"] if isinstance(out, dict) else str(out))
        out_ids = None
        obj = out if not isinstance(out, dict) else out
        for attr in ("output_ids", "output_token_ids", "token_ids"):
            v = getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)
            if v is not None:
                out_ids = v
                break
        print(f"[{MODE}] OUTPUT_TEXT={text!r}", flush=True)
        print(f"[{MODE}] OUTPUT_IDS={_to_id_list(out_ids)}", flush=True)

        regurg = text.lstrip().startswith(PROMPT.strip()) if isinstance(text, str) else None
        prompt_ids = None
        if tok is not None:
            try:
                prompt_ids = tok.encode(PROMPT)
                prompt_ids = prompt_ids["input_ids"] if isinstance(prompt_ids, dict) else prompt_ids
            except Exception:
                prompt_ids = None

        # ---- perf: best-of-3 ----
        times = []
        for i in range(3):
            t = time.time()
            _ = gen(MAX_NEW)
            dt = time.time() - t
            times.append(dt)
            print(f"[{MODE}] run {i}: {dt:.3f}s for {MAX_NEW} new tokens", flush=True)
        best = min(times)
        tpot_ms = best / MAX_NEW * 1000
        tps = MAX_NEW / best

        spec_info = None
        if MODE == "dflash":
            try:
                spec_info = str(engine.get_server_info())[:4000]
            except Exception as e:
                spec_info = f"get_server_info failed: {e}"

        result = {
            "mode": MODE,
            "prompt": PROMPT,
            "output_text": text,
            "output_ids": _to_id_list(out_ids),
            "prompt_ids": _to_id_list(prompt_ids),
            "regurgitates_prompt": bool(regurg) if regurg is not None else None,
            "max_new_tokens": MAX_NEW,
            "times_s": times,
            "best_time_s": best,
            "tpot_ms": tpot_ms,
            "tokens_per_s": tps,
            "spec_info": spec_info,
        }
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[{MODE}] SAVED {OUT}: tpot_ms={tpot_ms:.2f} tps={tps:.2f} regurg={result['regurgitates_prompt']}", flush=True)
        engine.shutdown()
        print(f"[{MODE}] DONE", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[{MODE}] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        with open(OUT + ".fail", "w") as f:
            f.write(f"{type(e).__name__}: {e}\n\n")
            traceback.print_exc(file=f)
        sys.exit(2)


if __name__ == "__main__":
    main()
