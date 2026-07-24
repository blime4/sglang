#!/usr/bin/env python3
# DL begin — test unequal-length multi-seq PREFILL (extend) via dl_flash_attn.
#
# Bug under test (dl_flash_attn.py):
#   The wrapper computed `_seqq = q.shape[0] // _batch` (AVERAGE query length)
#   and passed it as `max_seqlen_q`, discarding the correct `max_seqlen_q_arg`
#   (= max(extend_seq_lens)) the backend passes in.
#   - decode / equal-len prefill: avg == max, OK
#   - UNEQUAL-len multi-seq prefill (online concurrency): avg < max -> kernel
#     softmax_lse workspace under-sized -> OOB.
#
# This script verifies the FIX: it spies on _sgl_fa2_varlen and asserts the
# wrapper passes max_seqlen_q=max(extend_lens) and the correct cu_seqlens_q.
# (Numeric comparison is NOT run here — the DLIN dldnn paged multi-query prefill
# kernel SIGSEGVs in dleol JIT for this shape, independent of the wrapper. That
# is a separate issue to triage; numeric correctness is validated end-to-end.)
#
# Env: source $SDK_DIR/env.sh; CUDA_VISIBLE_DEVICES=<free>.
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.dl_flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
dev = "cuda"
B = 4
Hq, Hkv, D = 16, 8, 128      # GQA, head_dim 128
Pg = 16                       # page size (sglang DLIN default)
scale = 1.0 / (D ** 0.5)

extend_lens = [16, 8, 32, 3]   # UNEQUAL — the online-concurrency shape
assert len(extend_lens) == B
total_q = sum(extend_lens)     # 59
max_q = max(extend_lens)       # 32  <- CORRECT max_seqlen_q
avg_q = total_q // B           # 14  <- BUGGY value the wrapper used to infer

max_blocks_per_seq = (max_q + Pg - 1) // Pg    # 2
# exclusive physical pages per seq (no duplicates, like real sglang)
bt_rows, _phys = [], 0
for _b in range(B):
    bt_rows.append(list(range(_phys, _phys + max_blocks_per_seq)))
    _phys += max_blocks_per_seq
num_blocks = _phys
k_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
v_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
block_table = torch.tensor(bt_rows, dtype=torch.int32, device=dev)
cache_seqlens = torch.tensor(extend_lens, dtype=torch.int32, device=dev)
q_packed = torch.randn(total_q, Hq, D, dtype=torch.bfloat16, device=dev) * 0.3
cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=dev)
cu_seqlens_q[1:] = torch.tensor(extend_lens).cumsum(0)


def ref_attention():
    """Per-seq causal SDPA on gathered contiguous KV (ground truth, for reference)."""
    outs = []
    ratio = Hq // Hkv
    for b in range(B):
        L = extend_lens[b]
        kv = k_cache[block_table[b]].reshape(max_blocks_per_seq * Pg, Hkv, D)[:L]
        vv = v_cache[block_table[b]].reshape(max_blocks_per_seq * Pg, Hkv, D)[:L]
        qb = q_packed[cu_seqlens_q[b]:cu_seqlens_q[b + 1]]
        q4 = qb.permute(1, 0, 2).unsqueeze(0)
        k4 = kv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        v4 = vv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=scale)
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, 0)


def spy_call():
    """Intercept _sgl_fa2_varlen, record what the wrapper passes, return dummy."""
    import sglang.srt.layers.attention.sgl_flash_attn as _sglfa
    seen = {}
    _real = _sglfa.flash_attn_varlen_func

    def _spy(**kw):
        seen.update(kw)
        return torch.zeros(total_q, Hq, D, dtype=torch.bfloat16, device=dev)

    _sglfa.flash_attn_varlen_func = _spy
    try:
        flash_attn_with_kvcache(
            q_packed, k_cache, v_cache,
            page_table=block_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_q,
            softmax_scale=scale, causal=True,
        )
    finally:
        _sglfa.flash_attn_varlen_func = _real
    return seen


def main():
    print(f"B={B} extend_lens={extend_lens} total_q={total_q}")
    print(f"  CORRECT max_seqlen_q={max_q}  vs  BUGGY avg-derived _seqq={avg_q}")
    seen = spy_call()
    msmq = seen.get("max_seqlen_q")
    cuq = seen.get("cu_seqlens_q")
    msmq_ok = msmq == max_q
    cuq_ok = cuq is not None and cuq.tolist() == cu_seqlens_q.tolist()
    print(f"[spy] max_seqlen_q={msmq} (expect {max_q}) -> {'OK' if msmq_ok else 'FAIL'}")
    print(f"[spy] cu_seqlens_q={cuq.tolist() if cuq is not None else None} "
          f"(expect {cu_seqlens_q.tolist()}) -> {'OK' if cuq_ok else 'FAIL'}")
    ok = msmq_ok and cuq_ok
    print("RESULT:",
          "PASS — wrapper passes correct max_seqlen_q / cu_seqlens_q" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
# DL end
