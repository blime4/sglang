#!/usr/bin/env python3
# Validate the varlen-for-decode workaround for the DLIN dleol
# flash_attn_with_kvcache "to bc failed" bug.
#
# Idea: dleol's paged-DECODE kernel (flash_attn_with_kvcache) is broken, but its
# PREFILL kernel (flash_attn_varlen_func) works. So for decode, gather the paged
# KV cache into a packed (varlen) layout and call flash_attn_varlen_func.
#
# This script proves the gather+varlen result matches a torch reference for the
# decode pattern (1 query token per seq, variable KV lengths), WITHOUT needing
# the model or the broken decode kernel.
#
# Run with the clean DLIN env (SDK_DIR=.../sdk), CUDA_VISIBLE_DEVICES=<free>.
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"
B = 4            # decode batch
Hq, Hkv, D = 16, 8, 128   # Qwen3-1.7B-ish GQA
Pg = 16          # page/block size (matches sglang DLIN default)
# per-seq kv lengths (variable, decode: cache already holds 1..N tokens)
seqlens = torch.tensor([3, 1, Pg + 5, 2 * Pg + 3], dtype=torch.int32, device=dev)
max_blocks = (int(seqlens.max().item()) + Pg - 1) // Pg
num_blocks = 64
scale = 1.0 / (D ** 0.5)

# paged KV cache (random), q (1 token / seq)
k_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
v_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
# block_table: each seq's logical blocks -> physical block ids (0..num_blocks-1)
block_table = torch.randint(0, num_blocks, (B, max_blocks), dtype=torch.int32, device=dev)
q = torch.randn(B, 1, Hq, D, dtype=torch.bfloat16, device=dev) * 0.3  # [B, seqlen_q=1, Hq, D]


def ref_attention():
    """Ground truth: per-seq torch SDPA on gathered contiguous KV."""
    outs = []
    ratio = Hq // Hkv
    for b in range(B):
        s = int(seqlens[b].item())
        blk = block_table[b]  # [max_blocks]
        kv = k_cache[blk].reshape(max_blocks * Pg, Hkv, D)[:s]  # [s, Hkv, D]
        vv = v_cache[blk].reshape(max_blocks * Pg, Hkv, D)[:s]
        # SDPA shapes: [batch=1, heads, seq, dim]
        qb4 = q[b, 0].unsqueeze(1).unsqueeze(0)  # [1, Hq, 1, D]
        kv_e = kv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)  # [1, Hq, s, D]
        vv_e = vv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)  # [1, Hq, s, D]
        o = F.scaled_dot_product_attention(qb4, kv_e, vv_e, is_causal=False, scale=scale)
        outs.append(o.squeeze(0).transpose(0, 1).reshape(1, Hq, D))  # [1, Hq, D]
    return torch.cat(outs, 0)  # [B, Hq, D]


def varlen_decode():
    """Workaround: boolean-mask gather -> packed -> flash_attn_varlen_func."""
    # gather all blocks per seq: [B, max_blocks, Pg, Hkv, D] -> [B, M, Hkv, D]
    gathered_k = k_cache[block_table].reshape(B, max_blocks * Pg, Hkv, D)
    gathered_v = v_cache[block_table].reshape(B, max_blocks * Pg, Hkv, D)
    M = max_blocks * Pg
    pos = torch.arange(M, device=dev).unsqueeze(0)  # [1, M]
    mask = pos < seqlens.long().unsqueeze(1)  # [B, M]
    k_packed = gathered_k[mask]  # [total_k, Hkv, D]
    v_packed = gathered_v[mask]
    cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    cu_seqlens_k[1:] = seqlens.long().cumsum(0)
    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev)  # 1 q token/seq
    q_v = q.squeeze(1)  # [B, Hq, D]
    max_seqlen_k = int(seqlens.max().item())
    out = flash_attn_varlen_func(
        q_v, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=1, max_seqlen_k=max_seqlen_k, softmax_scale=scale, causal=False,
    )
    return out  # [B, Hq, D]


def main():
    print(f"B={B} Hq={Hq} Hkv={Hkv} D={D} Pg={Pg} seqlens={seqlens.tolist()} max_blocks={max_blocks}")
    ref = ref_attention().float()
    print("ref_attention OK", ref.shape)
    out = varlen_decode().float()
    print("varlen_decode OK", out.shape)
    max_err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
    print(f"varlen-decode vs torch SDPA: max_err={max_err:.4e}  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
