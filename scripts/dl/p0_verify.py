#!/usr/bin/env python3
"""P0 决定性验证：spec verify 垃圾输出假阳性是否已修 + accept 天花板。

一个问题、两组对照：
  (1) 垃圾输出假阳性是否已修？  -> PLAIN(非投机) 与 SPEC(FROZEN_KV_MTP) 在相同
      prompt + 相同采样下输出是否一致。若一致(都连贯) => spec 路径正确，accept 是真实值。
  (2) accept 天花板 = ?           -> 可预测 prompt(计数序列) vs 新颖 prompt(开放问答)。

Env: MODEL, TP_SIZE(2), PREDICTABLE, NOVEL, MAX_NEW_TOKENS
     (MoE env: SGLANG_DL_MOE_FUSED=1 FUSED_MAX_M=16 MAX_BF16_M=128 在 launcher 设置)
"""
import os, sys, time
import sglang
from sglang.srt.server_args import ServerArgs

MODEL = os.environ.get("MODEL", "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8")
TP = int(os.environ.get("TP_SIZE", "2"))
PREDICTABLE = os.environ.get("PREDICTABLE", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,")
NOVEL = os.environ.get("NOVEL", "Explain how neural networks learn from data.")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "96"))

SAMPLINGS = [
    ("greedy",          {"max_new_tokens": MAX_NEW, "temperature": 0}),
    ("greedy+rep1.2",   {"max_new_tokens": MAX_NEW, "temperature": 0, "repetition_penalty": 1.2}),
    ("sample0.6",       {"max_new_tokens": MAX_NEW, "temperature": 0.6}),
]

def make_engine(spec: bool):
    kw = dict(model_path=MODEL, dtype="bfloat16", tp_size=TP,
              attention_backend="fa3", page_size=16, mem_fraction_static=0.60,
              disable_cuda_graph=True, context_length=4096)
    if spec:
        kw.update(speculative_algorithm="FROZEN_KV_MTP", speculative_eagle_topk=1,
                  speculative_num_steps=4, speculative_num_draft_tokens=5)
    return sglang.Engine(server_args=ServerArgs(**kw))

def gen(e, prompt, sp, label):
    t = time.time()
    r = e.generate(prompt, sampling_params=sp)
    dt = time.time() - t
    m = r["meta_info"]; n = m["completion_tokens"]
    spec = {k: v for k, v in m.items() if "spec" in k.lower() or "accept" in k.lower()}
    print(f"  [{label}] {n} tok / {dt:.2f}s = {n/dt:.2f} tok/s | {spec}")
    print(f"    text: {r['text'][:300]!r}")
    return r["text"], spec, n / dt

def main():
    prompts = [("PREDICTABLE", PREDICTABLE), ("NOVEL", NOVEL)]
    print(f"MODEL={MODEL} TP={TP} MAX_NEW={MAX_NEW}", flush=True)

    # --- plain reference ---
    t0 = time.time(); e = make_engine(spec=False); print(f"[plain] loaded {time.time()-t0:.1f}s", flush=True)
    e.generate(NOVEL, sampling_params={"max_new_tokens": 16, "temperature": 0})  # warmup
    plain = {}
    for pname, p in prompts:
        plain[pname] = {}
        for sname, sp in SAMPLINGS:
            txt, _, _ = gen(e, p, sp, f"PLAIN {pname}/{sname}")
            plain[pname][sname] = txt
    e.shutdown(); del e
    print("[plain] done\n", flush=True)

    # --- spec ---
    t0 = time.time(); e = make_engine(spec=True); print(f"[spec]  loaded {time.time()-t0:.1f}s", flush=True)
    e.generate(NOVEL, sampling_params={"max_new_tokens": 16, "temperature": 0})
    e.generate(NOVEL, sampling_params={"max_new_tokens": 16, "temperature": 0, "repetition_penalty": 1.2})
    rows = []
    for pname, p in prompts:
        for sname, sp in SAMPLINGS:
            txt, specm, tps = gen(e, p, sp, f"SPEC  {pname}/{sname}")
            ptxt = plain[pname][sname]
            match = (txt[:80] == ptxt[:80])
            rows.append((pname, sname, match, specm, tps, txt, ptxt))
    e.shutdown(); del e

    print("\n" + "=" * 70)
    print("P0 SUMMARY: SPEC vs PLAIN (first-80-char match = spec path correct)")
    print("=" * 70)
    for pname, sname, match, specm, tps, txt, ptxt in rows:
        flag = "MATCH" if match else "DIFF "
        acc = specm.get("spec_accept_rate", specm.get("accept_rate", specm.get("draft_accept_rate", "?")))
        acl = specm.get("spec_accept_length", specm.get("accept_length", "?"))
        print(f"  {pname:11s} {sname:13s} {flag} accept={acc} len={acl} | spec {tps:.1f} tok/s")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
