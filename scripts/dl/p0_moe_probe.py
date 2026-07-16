#!/usr/bin/env python3
"""P0 残留根因探针：verify forward ≠ decode forward。隔离 fused MoE M>1 嫌疑。

跑 spec(FROZEN_KV_MTP,topk=1) 的 NOVEL/greedy，扫 SGLANG_DL_MOE_FUSED_MAX_M，
打印输出 + accept。对照 plain greedy 基线("\n\n\n...")：
  - 若某 M 值下 spec 输出 ≈ plain("\n\n\n") => fused MoE M>1 是残留根因，可落地修。
  - 若所有 M 都偏离 plain => 根因在别处(GDN 状态ful batch≠seq / attention)。
"""
import os, sys, time
import sglang
from sglang.srt.server_args import ServerArgs

MODEL = "/LocalRun/xi.chen/Qwen3.5-35B-A3B-FP8"
NOVEL = "Explain how neural networks learn from data."
TP = 2

def main():
    e = sglang.Engine(server_args=ServerArgs(
        model_path=MODEL, dtype="bfloat16", tp_size=TP, attention_backend="fa3",
        page_size=16, mem_fraction_static=0.60, disable_cuda_graph=True, context_length=4096,
        speculative_algorithm="FROZEN_KV_MTP", speculative_eagle_topk=1,
        speculative_num_steps=4, speculative_num_draft_tokens=5))
    e.generate(NOVEL, sampling_params={"max_new_tokens": 16, "temperature": 0})  # warmup
    print(f"\nFUSED_MAX_M={os.environ.get('SGLANG_DL_MOE_FUSED_MAX_M','?')} MAX_BF16_M={os.environ.get('SGLANG_DL_MOE_MAX_BF16_M','?')}", flush=True)
    for trial in range(2):
        t = time.time()
        r = e.generate(NOVEL, sampling_params={"max_new_tokens": 96, "temperature": 0})
        dt = time.time() - t
        m = r["meta_info"]
        acc = {k: v for k, v in m.items() if "accept" in k.lower()}
        print(f"[trial{trial}] {m['completion_tokens']}tok/{dt:.1f}s accept={acc}", flush=True)
        print(f"  text: {r['text'][:200]!r}", flush=True)
    e.shutdown()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush(); sys.exit(1)
