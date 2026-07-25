#!/usr/bin/env python3
# DL: APC probe — can vLLM prefix caching (APC) be enabled on the hybrid-Mamba
# Qwen3.5/3.6-35B-A3B model, in a STABLE config? Defuses the demo critique "you
# disabled APC so sglang wins."
#
# Tests BOTH runners explicitly (an earlier version forgot to set
# VLLM_USE_V2_MODEL_RUNNER, silently using the MRV1 default and misleadingly
# 'succeeding'). Findings (build v0.21.1.dev2+g9511db443):
#   - MRV2 (the compare's stable runner) + APC: FAILS — "Model Runner V2 has not
#     yet supported mamba_cache_mode='align'" (vllm/config/vllm.py:2030). Under CG
#     it instead hits block_size(528) > max_num_batched_tokens(4). Either way FAIL.
#   - MRV1 + APC + eager: INITS + correct output — but MRV1 is unstable on DLIN
#     (the compare skips it: crashes ConstraintViolationError / assert).
# => APC cannot be enabled in vLLM's STABLE DLIN config (MRV2). APC-OFF is FORCED.
#    The compare is fair (each engine's only stable working config).
#
# Run: source sdk-dlop-07-13-20-30/env.sh; CUDA_VISIBLE_DEVICES=<free> .venv/bin/python scripts/dl/apc_failure_probe.py
import os, sys, time

MODEL = os.environ.get(
    "MODEL_PATH",
    "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/models/Qwen3.6-35B-A3B-FP8",
)
TP = int(os.environ.get("TP_SIZE", "4"))


def try_runner_apc(runner):
    """Construct vLLM LLM with APC under the given runner. Returns (llm, status, detail)."""
    from vllm import LLM, SamplingParams  # noqa: F401  (SamplingParams used by caller)
    mrv2 = (runner == "mrv2")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if mrv2 else "0"
    kw = dict(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.55,
              trust_remote_code=True, max_num_seqs=4, disable_log_stats=True,
              enable_prefix_caching=True)
    # MRV2+APC fails under CG (block_size) AND eager (align-unsupported); use CG to
    # show the compare's exact config. MRV1 must be eager (compile crash).
    if mrv2:
        kw["enforce_eager"] = False
        kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4],
                                    "max_cudagraph_capture_size": 4}
    else:
        kw["enforce_eager"] = True
    llm = LLM(**kw)
    return llm


if __name__ == "__main__":
    print(f"[apc-probe] model={MODEL} tp={TP}", flush=True)
    results = {}
    for runner in ["mrv2", "mrv1"]:
        print(f"\n[apc-probe] === {runner} + enable_prefix_caching=True ===", flush=True)
        llm = None
        try:
            llm = try_runner_apc(runner)
            # init succeeded -> correctness check (long gen past <think>)
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0, max_tokens=300, ignore_eos=True)
            prefix = ("You are a precise assistant. Answer in a few words.\n\n"
                      "Q: What is 2+2? A: 4\nQ: Capital of France? A: Paris\n\n"
                      "Document: The KS38 has 192 FP8 TFLOPS per QUAD and 4 QUADs per board.\n\n")
            o1 = llm.generate([prefix + "Q: FP8 throughput per QUAD? A:"], sp)[0].outputs[0].text
            o2 = llm.generate([prefix + "Q: QUADs per board? A:"], sp)[0].outputs[0].text
            ok = ("192" in o1) and ("4" in o2) and (o1.strip() != o2.strip())
            results[runner] = ("WORKS", f"correct={ok} (p1->192={('192' in o1)}, p2->4={('4' in o2)})")
            print(f"[apc-probe] {runner}: APC init OK, correct={ok}", flush=True)
        except Exception as e:  # noqa: BLE001
            etype = type(e).__name__; msg = str(e)[:200].replace("\n", " ")
            results[runner] = ("FAILS", f"{etype}: {msg}")
            print(f"[apc-probe] {runner}: APC init FAILS ({etype}): {msg}", flush=True)
        finally:
            del llm

    print("\n=== APC PROBE SUMMARY ===", flush=True)
    for r, (s, d) in results.items():
        print(f"  {r}: {s} — {d}", flush=True)
    mrv2_works = results.get("mrv2", ("FAILS",))[0] == "WORKS"
    if mrv2_works:
        print("VERDICT: !! MRV2+APC WORKS — APC-OFF is a CHOICE; re-run compare with "
              "MRV2+APC before demoing!", flush=True)
        sys.exit(1)
    else:
        print("VERDICT: MRV2 (vLLM's stable DLIN runner) CANNOT enable APC. MRV1 can "
              "(eager) but is unstable on DLIN. => APC-OFF is FORCED for the stable "
              "config; the compare is FAIR.", flush=True)
        sys.exit(0)
