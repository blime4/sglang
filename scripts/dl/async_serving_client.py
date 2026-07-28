#!/usr/bin/env python3
"""True-async serving throughput client (aiohttp) — rules out Python-thread/GIL artifacts.
Fires C requests SIMULTANEOUSLY via asyncio.gather over a pooled ClientSession. Same
shared-prefix workload as exp_b_serving_client.py.
"""
import argparse, asyncio, json, time, statistics
import aiohttp

SHARED = (
    "You are a senior staff engineer writing an internal postmortem. Here is the long "
    "incident timeline and telemetry context that every question below refers to: "
    "At 03:11 UTC the ingestion pipeline latency crossed the SLO band. The scheduler "
    "had been batching prefill and decode across four tensor-parallel workers. KV cache "
    "was managed via a radix tree so that shared system prefixes were computed once. "
)
QUESTIONS = [
    "What was the root cause of the latency spike?", "Which kernel path caused the failure?",
    "How many experts are selected per token?", "What does the radix tree cache reuse?",
    "What quantization format is the weight in?", "How many TP workers were used?",
    "When did latency cross the SLO band?", "What is the act-quant step dependent on?",
    "Summarize the incident in two sentences.", "Name the batching strategy used.",
] * 8

def build_prefix(t):
    return (SHARED + "\n") * max(1, t // 120)

async def one(session, url, model, prompt, max_tokens):
    body = json.dumps({"model": model, "prompt": prompt, "temperature": 0.0,
                       "max_tokens": max_tokens, "ignore_eos": True}).encode()
    t0 = time.perf_counter()
    async with session.post(url + "/completions", data=body,
                            headers={"Content-Type": "application/json"}) as r:
        obj = await r.json()
    return time.perf_counter() - t0, obj.get("usage", {}).get("completion_tokens", max_tokens)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", default="1,8,32")
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--prefix-tokens", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    prefix = build_prefix(args.prefix_tokens)
    print(f"[async] url={args.url} prefix~{args.prefix_tokens}tok max={args.max_tokens} waves={args.waves}", flush=True)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)  # unlimited connections
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # warmup
        try:
            await one(session, args.url, args.model, prefix + "\n" + QUESTIONS[0], 8)
            print("[async] warmup ok", flush=True)
        except Exception as e:
            print(f"[async] warmup FAILED: {e}", flush=True)
        print(f"{'conc':>5} {'best_aggregate_tps':>20} {'mean_lat_s':>10}", flush=True)
        for c in [int(x) for x in args.conc.split(",")]:
            best = 0.0; lats = []
            for w in range(args.waves):
                prompts = [prefix + "\nQ%d: " % (c*w+i) + QUESTIONS[(c*w+i) % len(QUESTIONS)] for i in range(c)]
                t0 = time.perf_counter()
                outs = await asyncio.gather(*[one(session, args.url, args.model, p, args.max_tokens) for p in prompts])
                wall = time.perf_counter() - t0
                tot = sum(o[1] for o in outs); tps = tot / wall; lats.extend(o[0] for o in outs)
                best = max(best, tps)
            print(f"{c:>5} {best:>20.1f} {statistics.mean(lats):>10.2f}", flush=True)
    print("[async] DONE", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
