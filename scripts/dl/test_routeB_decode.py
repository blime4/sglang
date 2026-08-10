#!/usr/bin/env python3
# Validate Route B: a graph-SAFE decode attention that works around the DLIN
# dleol flash_attn_with_kvcache crash WITHOUT breaking cuda-graph capture.
#
# Idea: cuda graph needs STATIC tensor shapes. The varlen workaround packs KV
# into a variable-size [total_k, ...] tensor -> not capturable. Route B instead
# gathers ALL pages into a FIXED [B, max_blocks*Pg, Hkv, D] tensor, masks the
# invalid tail (pos >= seqlen) by setting K to a large-negative (softmax zeroes
# it), and calls flash_attn_func (FA2, fixed-shape kernel launch -> capturable).
#
# This script proves Route B matches torch SDPA for the decode pattern.
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

torch.manual_seed(0)
dev = "cuda"
B = 4
Hq, Hkv, D = 16, 8, 128     # Qwen3-1.7B GQA
Pg = 16
seqlens = torch.tensor([3, 1, Pg + 5, 2 * Pg + 3], dtype=torch.int32, device=dev)
max_blocks = (int(seqlens.max().item()) + Pg - 1) // Pg
num_blocks = 64
scale = 1.0 / (D ** 0.5)
MASK_VAL = -1.0e4

k_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
v_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
block_table = torch.randint(0, num_blocks, (B, max_blocks), dtype=torch.int32, device=dev)
q = torch.randn(B, 1, Hq, D, dtype=torch.bfloat16, device=dev) * 0.3   # [B, 1, Hq, D]


def ref_attention():
    """Ground truth: per-seq torch SDPA on gathered contiguous KV."""
    outs = []
    ratio = Hq // Hkv
    for b in range(B):
        s = int(seqlens[b].item())
        kv = k_cache[block_table[b]].reshape(max_blocks * Pg, Hkv, D)[:s]
        vv = v_cache[block_table[b]].reshape(max_blocks * Pg, Hkv, D)[:s]
        qb4 = q[b, 0].unsqueeze(1).unsqueeze(0)                       # [1, Hq, 1, D]
        kv_e = kv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        vv_e = vv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        o = F.scaled_dot_product_attention(qb4, kv_e, vv_e, is_causal=False, scale=scale)
        outs.append(o.squeeze(0).transpose(0, 1).reshape(1, Hq, D))
    return torch.cat(outs, 0)                                          # [B, Hq, D]


def route_b_decode():
    """Graph-SAFE: fixed-shape gather + mask invalid K + flash_attn_func."""
    M = max_blocks * Pg
    gk = k_cache[block_table].reshape(B, M, Hkv, D)      # [B, M, Hkv, D] fixed
    gv = v_cache[block_table].reshape(B, M, Hkv, D)
    pos = torch.arange(M, device=dev).unsqueeze(0)       # [1, M]
    invalid = pos >= seqlens.long().unsqueeze(1)         # [B, M]
    # mask invalid K positions so q.k -> large negative -> exp -> 0
    gk = gk.masked_fill(invalid.unsqueeze(-1).unsqueeze(-1), MASK_VAL)
    out = flash_attn_func(q, gk, gv, softmax_scale=scale, causal=False)  # [B,1,Hq,D]
    return out.squeeze(1)                                # [B, Hq, D]


def main():
    print(f"B={B} Hq={Hq} Hkv={Hkv} D={D} Pg={Pg} "
          f"seqlens={seqlens.tolist()} max_blocks={max_blocks} M={max_blocks*Pg}")
    ref = ref_attention().float()
    out = route_b_decode().float()
    max_err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
    print(f"Route B (flash_attn_func + fixed-shape gather + K-mask) vs torch SDPA: "
          f"max_err={max_err:.4e}  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
