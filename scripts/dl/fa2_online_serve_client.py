#!/usr/bin/env python3
# DL: FA2 online-concurrency SERVING test — concurrent unequal-length HTTP requests
# against a running sglang server (fa3 = DLIN FA2). Complements
# test_fa2_online_concurrency.py (offline batch) by exercising the real ONLINE
# serving path: requests arrive concurrently, their unequal-length prefills
# overlap inside the scheduler (one varlen extend), the pre-fix bug shape.
#
# Prereq: server up, e.g.
#   CUDA_VISIBLE_DEVICES=20,21,22,23 ./run_sglang.sh serve -M qwen35-35b --port 30000
import sys, time, json, urllib.request, concurrent.futures


def post(url, payload, timeout=180):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
    comp = base + "/v1/completions"
    try:
        # /v1/models is a GET (post() would 405)
        models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())
        name = models["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        name = "default"
        print(f"[fa2-online] /v1/models failed ({e}); using model={name}", flush=True)
    print(f"[fa2-online] server={base} model={name}", flush=True)

    pbase = ("Explain in detail the architecture of a heterogeneous AI accelerator: "
             "compute units, memory hierarchy, programming model, and tradeoffs. ")
    prompts = [pbase * m + "\n\nSummarize the key point in one sentence."
               for m in [1, 4, 10, 2, 7, 3, 5, 8]]  # UNEQUAL lengths

    # warmup (also proves the server responds before the concurrent burst)
    post(comp, {"model": name, "prompt": prompts[0], "temperature": 0,
                "max_tokens": 8, "ignore_eos": True})

    def one(p):
        return post(comp, {"model": name, "prompt": p, "temperature": 0,
                           "max_tokens": 16, "ignore_eos": True})["choices"][0]["text"]

    n = len(prompts)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        outs = list(ex.map(one, prompts))
    dt = time.perf_counter() - t0
    ok = len(outs) == n and all(o.strip() for o in outs)
    print(f"[fa2-online] N={n} concurrent UNEQUAL-len requests: {dt:.2f}s all_nonempty={ok}",
          flush=True)
    for i, o in enumerate(outs):
        print(f"  [{i}] {o[:60]!r}", flush=True)
    print("VERDICT:", "PASS — online server handles concurrent unequal-length prefill "
          "(no crash, valid output)" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
