#!/usr/bin/env python3
"""Diagnostic script for dldnn FA2 crash on unequal-length multi-sequence prefill.

Spawns subprocesses to test different cases and captures exit codes:
- A: B=1, single-sequence prefill
- B: B=4, equal-length multi-sequence prefill
- C: B=4, unequal-length multi-sequence prefill (known crash case)
- D: B=4, decode (single query token per sequence, should work)

Each case runs in its own subprocess because SIGSEGV kills the process.
"""

import os
import sys
import subprocess
import argparse


def run_case(case: str) -> tuple[int, str, str]:
    """Run a single case in a subprocess. Returns (exit_code, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang/python:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, __file__, "--case", case],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,  # 60 second timeout per case
    )
    return result.returncode, result.stdout, result.stderr


def run_case_A():
    """Case A: B=1, single-sequence prefill (32 tokens)"""
    import torch
    from sglang.srt.layers.attention.dl_flash_attn import flash_attn_with_kvcache

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Setup: B=1, Hq=16, Hkv=8, D=128, page_size=16
    B, Hq, Hkv, D, Pg = 1, 16, 8, 128, 16
    extend_lens = [32]

    # q: (B, max_extend_len, Hq, D) -> (1, 32, 16, 128)
    max_extend = max(extend_lens)
    q = torch.randn(B, max_extend, Hq, D, dtype=dtype, device=device)

    # k_cache, v_cache: (num_blocks, page_size, Hkv, D)
    # Allocate enough blocks for all sequences, no duplicate pages
    max_blocks = (max(extend_lens) + Pg - 1) // Pg
    k_cache = torch.randn(max_blocks, Pg, Hkv, D, dtype=dtype, device=device)
    v_cache = torch.randn(max_blocks, Pg, Hkv, D, dtype=dtype, device=device)

    # block_table: (B, max_blocks_per_seq)
    block_table = torch.arange(0, max_blocks, dtype=torch.int32, device=device).unsqueeze(0).expand(B, max_blocks)

    # cache_seqlens: (B,) - current KV cache length (0 for new prefill)
    cache_seqlens = torch.tensor([0] * B, dtype=torch.int32, device=device)

    # cu_seqlens_q: cumulative sequence lengths for q
    cu_seqlens_q = torch.tensor([0, extend_lens[0]], dtype=torch.int32, device=device)

    # max_seqlen_q: maximum query sequence length
    max_seqlen_q = max(extend_lens)

    print(f"Case A: B={B}, extend_lens={extend_lens}, max_seqlen_q={max_seqlen_q}")

    try:
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=None,
            causal=True,
        )
        print(f"  OK: output shape {out.shape if out is not None else 'None'}")
        return 0
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return 1


def run_case_B():
    """Case B: B=4, equal-length multi-sequence prefill (16 tokens each)"""
    import torch
    from sglang.srt.layers.attention.dl_flash_attn import flash_attn_with_kvcache

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Setup: B=4, Hq=16, Hkv=8, D=128, page_size=16
    B, Hq, Hkv, D, Pg = 4, 16, 8, 128, 16
    extend_lens = [16, 16, 16, 16]

    # q: (B, max_extend_len, Hq, D) -> (4, 16, 16, 128)
    max_extend = max(extend_lens)
    q = torch.randn(B, max_extend, Hq, D, dtype=dtype, device=device)

    # k_cache, v_cache: each sequence gets its own blocks (no duplicate pages)
    # Need 1 block per sequence (16 tokens / 16 page_size = 1)
    max_blocks_per_seq = (max_extend + Pg - 1) // Pg
    total_blocks = B * max_blocks_per_seq
    k_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)
    v_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)

    # block_table: (B, max_blocks_per_seq)
    # Each sequence gets its own block range: [0], [1], [2], [3]
    block_table = torch.arange(0, total_blocks, dtype=torch.int32, device=device).view(B, max_blocks_per_seq)

    # cache_seqlens: (B,) - all 0 for new prefill
    cache_seqlens = torch.tensor([0] * B, dtype=torch.int32, device=device)

    # cu_seqlens_q: cumulative sequence lengths for q (ensure int32 after cumsum)
    cu_seqlens_q = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)

    # max_seqlen_q: maximum query sequence length
    max_seqlen_q = max(extend_lens)

    print(f"Case B: B={B}, extend_lens={extend_lens}, max_seqlen_q={max_seqlen_q}")

    try:
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=None,
            causal=True,
        )
        print(f"  OK: output shape {out.shape if out is not None else 'None'}")
        return 0
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return 1


def run_case_C():
    """Case C: B=4, unequal-length multi-sequence prefill (16, 8, 32, 3) - KNOWN CRASH"""
    import torch
    from sglang.srt.layers.attention.dl_flash_attn import flash_attn_with_kvcache

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Setup: B=4, Hq=16, Hkv=8, D=128, page_size=16
    B, Hq, Hkv, D, Pg = 4, 16, 8, 128, 16
    extend_lens = [16, 8, 32, 3]

    # q: (B, max_extend_len, Hq, D) -> (4, 32, 16, 128)
    max_extend = max(extend_lens)
    q = torch.randn(B, max_extend, Hq, D, dtype=dtype, device=device)

    # k_cache, v_cache: allocate enough blocks for the longest sequence
    max_blocks_per_seq = (max_extend + Pg - 1) // Pg
    total_blocks = B * max_blocks_per_seq
    k_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)
    v_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)

    # block_table: (B, max_blocks_per_seq)
    # Each sequence gets its own block range
    block_table = torch.arange(0, total_blocks, dtype=torch.int32, device=device).view(B, max_blocks_per_seq)

    # cache_seqlens: (B,) - all 0 for new prefill
    cache_seqlens = torch.tensor([0] * B, dtype=torch.int32, device=device)

    # cu_seqlens_q: cumulative sequence lengths for q (ensure int32 after cumsum)
    cu_seqlens_q = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)

    # max_seqlen_q: maximum query sequence length
    max_seqlen_q = max(extend_lens)

    print(f"Case C: B={B}, extend_lens={extend_lens}, max_seqlen_q={max_seqlen_q}")

    try:
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=None,
            causal=True,
        )
        print(f"  OK: output shape {out.shape if out is not None else 'None'}")
        return 0
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return 1


def run_case_D():
    """Case D: B=4, decode (1 query token per sequence, cache has 16 tokens)"""
    import torch
    from sglang.srt.layers.attention.dl_flash_attn import flash_attn_with_kvcache

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Setup: B=4, Hq=16, Hkv=8, D=128, page_size=16
    B, Hq, Hkv, D, Pg = 4, 16, 8, 128, 16

    # Decode: 1 query token per sequence
    # q: (B, 1, Hq, D) -> (4, 1, 16, 128)
    q = torch.randn(B, 1, Hq, D, dtype=dtype, device=device)

    # k_cache, v_cache: each sequence has 1 block with 16 cached tokens
    max_blocks_per_seq = 1
    total_blocks = B * max_blocks_per_seq
    k_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)
    v_cache = torch.randn(total_blocks, Pg, Hkv, D, dtype=dtype, device=device)

    # block_table: (B, max_blocks_per_seq)
    block_table = torch.arange(0, total_blocks, dtype=torch.int32, device=device).view(B, max_blocks_per_seq)

    # cache_seqlens: (B,) - each has 16 tokens cached
    cache_seqlens = torch.tensor([16] * B, dtype=torch.int32, device=device)

    # cu_seqlens_q: cumulative sequence lengths for q (1 each)
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=device)

    # max_seqlen_q: 1 for decode
    max_seqlen_q = 1

    print(f"Case D: B={B}, decode, cache_seqlens=[16]*{B}, max_seqlen_q={max_seqlen_q}")

    try:
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=None,
            causal=True,
        )
        print(f"  OK: output shape {out.shape if out is not None else 'None'}")
        return 0
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return 1


CASE_FUNCS = {
    "A": run_case_A,
    "B": run_case_B,
    "C": run_case_C,
    "D": run_case_D,
}

CASE_DESCRIPTIONS = {
    "A": "B=1 prefill (single seq, 32 tokens)",
    "B": "B=4 equal prefill (16 tok/seq)",
    "C": "B=4 unequal prefill (16,8,32,3) - KNOWN CRASH",
    "D": "B=4 decode (1 tok/seq, cache=16)",
}


def main():
    parser = argparse.ArgumentParser(description="Diagnose dldnn FA2 crash")
    parser.add_argument("--case", choices=list(CASE_FUNCS.keys()), help="Run specific case (for subprocess)")
    args = parser.parse_args()

    # If --case is specified, run that case directly (for subprocess)
    if args.case:
        sys.exit(CASE_FUNCS[args.case]())

    # Otherwise, run all cases as parent process
    print("=== dldnn FA2 Crash Diagnostic ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()

    results = {}
    for case in ["A", "B", "C", "D"]:
        print(f"Running case {case} ({CASE_DESCRIPTIONS[case]})...", flush=True)
        exit_code, stdout, stderr = run_case(case)
        results[case] = (exit_code, stdout, stderr)

        # Determine status
        if exit_code == 0:
            status = "OK"
        elif exit_code == 139:
            status = "CRASH (SIGSEGV)"
        elif exit_code == 134:
            status = "CRASH (SIGABRT)"
        elif exit_code == 255:
            status = "CRASH (unknown)"
        else:
            status = f"FAIL (exit={exit_code})"

        print(f"case {case}: exit={exit_code}  {status}")

        # Show stderr head if crash
        if exit_code not in (0, 1):
            stderr_head = stderr.strip().split("\n")[:5]
            if stderr_head and stderr_head[0]:
                print(f"  stderr head: {stderr_head[0]}")

        print(f"  output: {stdout.strip()[:200]}")
        print()

    # Summary
    print("=" * 60)
    print("## 判断 (Diagnosis)")
    print()

    crashes = [c for c, (code, _, _) in results.items() if code not in (0, 1)]

    if not crashes:
        print("No crashes detected - all cases passed!")
    else:
        print(f"Crashed cases: {', '.join(crashes)}")

        # Analyze pattern
        if "A" in crashes and "B" not in crashes:
            print("- Pattern: Single-sequence prefill crashes, multi-seq OK → weird!")
        elif "C" in crashes and "A" not in crashes and "B" not in crashes:
            print("- Pattern: Unequal-length multi-sequence prefill crashes")
            print("  → Crash trigger = UNEQUAL LENGTHS + MULTI-SEQUENCE")
        elif "A" in crashes and "C" in crashes:
            print("- Pattern: All prefill crashes, decode OK")
            print("  → Crash trigger = MULTI-QUERY (_seqq > 1)")
        elif "D" in crashes:
            print("- Pattern: Even decode crashes → FA2 completely broken on dldnn")
        else:
            print(f"- Pattern: Inconclusive, crashed cases: {crashes}")

    print()
    print("=" * 60)
    print("## 含义 (Implications)")

    if "C" in crashes and "A" not in crashes and "B" not in crashes:
        print("FA2-varlen 等长多序列 OK，但不等长崩 →")
        print("  可能是 dldnn MHAVarlenForward 对 cu_seqlens_q 不等长")
        print("  的形状处理有 bug。wrapper 的 max_seqlen_q 修复可能不够。")
    elif "B" in crashes and "A" not in crashes:
        print("单序列 OK，多序列即崩 →")
        print("  可能是 dldnn 对 batch > 1 的处理有问题。")
    elif all(c in crashes for c in ["A", "B", "C"]) and "D" not in crashes:
        print("所有 prefill 崩，只有 decode OK →")
        print("  dldnn FA2 在线并发（多 query token）被阻塞，")
        print("  只能用于 decode。")
    elif "D" in crashes:
        print("连 decode 都崩 → dldnn FA2 完全不可用。")
    else:
        print("需要更多信息才能确定。")

    return 0


if __name__ == "__main__":
    main()
