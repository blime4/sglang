#!/usr/bin/env python3
"""Exp B client: concurrent shared-prefix serving throughput.

Hits an OpenAI /v1/completions endpoint (sglang server OR vLLM server) with waves
of C concurrent requests that all SHARE a long system/document prefix + a unique
short suffix. Measures aggregate output tok/s = total_output_tokens / wave_wall_time.

Usage:
  python exp_b_serving_client.py --url http://127.0.0.1:30000/v1 --model default \
      --conc 1,4,8,16,32 --waves 3 --prefix-tokens 1500 --max-tokens 64

All clients share the prefix => exercises RadixAttention (sglang) / APC (vLLM) cache
reuse + concurrent decode batching = sglang's design center.
"""
import argparse, json, statistics, time, urllib.request, urllib.error, concurrent.futures as cf

SHARED = (
    "You are a senior staff engineer writing an internal postmortem. Here is the long "
    "incident timeline and telemetry context that every question below refers to: "
    "At 03:11 UTC the ingestion pipeline latency crossed the SLO band. The scheduler "
    "had been batching prefill and decode across four tensor-parallel workers. KV cache "
    "was managed via a radix tree so that shared system prefixes were computed once. "
    "The mixture-of-experts layer selected two experts per token and dispatched their "
    "weights into a fused grouped GEMM. Eight-bit floating-point quantization halved "
    "the weight memory but the act-quant step required a capture-safe kernel path. "
)  # ~120 tokens; repeat to reach prefix-tokens


def build_prefix(target_tokens):
    reps = max(1, target_tokens // 120)
    return (SHARED + "\n") * reps


def one_request(url, model, prompt, max_tokens, qid):
    body = json.dumps({"model": model, "prompt": prompt,
                       "temperature": 0.0, "max_tokens": max_tokens,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(url + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        obj = json.loads(r.read())
    dt = time.perf_counter() - t0
    out_tok = obj.get("usage", {}).get("completion_tokens", max_tokens)
    return dt, out_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base url, e.g. http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--conc", default="1,4,8,16,32")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--prefix-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    prefix = build_prefix(args.prefix_tokens)
    questions = [
        "What was the root cause of the latency spike?",
        "Which kernel path caused the capture failure?",
        "How many experts are selected per token here?",
        "What does the radix tree cache reuse?",
        "What quantization format is the weight in?",
        "How many tensor-parallel workers were used?",
        "When did latency cross the SLO band?",
        "What does the fused grouped GEMM operate on?",
        "Which step needed a capture-safe kernel?",
        "What halved the weight memory footprint?",
        "Summarize the incident in two sentences.",
        "Name the batching strategy the scheduler used.",
        "What is the act-quant step dependent on?",
        "What time did the issue begin (UTC)?",
        "What is the SLO band referenced here?",
        "Which layer selects two experts per token?",
    ] * 4

    print(f"[expB] url={args.url} model={args.model} prefix~{args.prefix_tokens}tok "
          f"max_tokens={args.max_tokens} waves={args.waves}", flush=True)
    # warmup: populate the shared-prefix cache
    try:
        one_request(args.url, args.model, prefix + "\n" + questions[0], 8, -1)
        print("[expB] warmup ok", flush=True)
    except Exception as e:
        print(f"[expB] warmup FAILED: {e}", flush=True)

    print(f"\n{'conc':>5} {'best_aggregate_tps':>20} {'mean_per_req_lat_s':>20} "
          f"{'best_wave_total_tok':>20}", flush=True)
    results = {}
    for c in [int(x) for x in args.conc.split(",")]:
        best_tps = 0.0
        best_tot = 0
        lats = []
        for w in range(args.waves):
            prompts = [prefix + "\nQ%d: " % (c * w + i) + questions[(c * w + i) % len(questions)]
                       for i in range(c)]
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(one_request, args.url, args.model, p, args.max_tokens, i)
                        for i, p in enumerate(prompts)]
                outs = [f.result() for f in futs]
            wall = time.perf_counter() - t0
            tot_tok = sum(o[1] for o in outs)
            tps = tot_tok / wall
            lats.extend(o[0] for o in outs)
            if tps > best_tps:
                best_tps, best_tot = tps, tot_tok
        mean_lat = statistics.mean(lats)
        results[c] = (best_tps, mean_lat, best_tot)
        print(f"{c:>5} {best_tps:>20.1f} {mean_lat:>20.3f} {best_tot:>20}", flush=True)
    print("\n[expB] DONE", flush=True)
    return results


if __name__ == "__main__":
    main()
