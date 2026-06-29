#!/usr/bin/env python3
# Validate a GRAPH-SAFE replacement for the boolean-index gather
# (`_gk[_mask]`, which is forbidden under cuda-graph capture because its output
# size is data-dependent). Idea: pack the valid per-seq KV into a FIXED-size
# buffer via a fixed-shape `scatter_` (destination indices = cumsum(seqlens)+pos,
# invalid positions routed to a never-read dummy slot). All ops are fixed-shape
# with runtime values -> cuda-graph-capturable. Then plain flash_attn_varlen_func
# on the fixed packed buffer (max_seqlen_k = upper bound, no .item() host-sync).
#
# This test: (a) scatter-packed == boolean-gather-packed (valid portion),
# (b) flash_attn_varlen_func(scatter-packed) == torch SDPA.
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"
B, Hq, Hkv, D, Pg = 4, 16, 8, 128, 16
seqlens = torch.tensor([3, 1, Pg + 5, 2 * Pg + 3], dtype=torch.int32, device=dev)
max_blocks = (int(seqlens.max().item()) + Pg - 1) // Pg
num_blocks = 64
scale = 1.0 / (D ** 0.5)
M = max_blocks * Pg
MAX_TOTAL = B * M

k_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
v_cache = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
block_table = torch.randint(0, num_blocks, (B, max_blocks), dtype=torch.int32, device=dev)
q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=dev) * 0.3  # [B, Hq, D], 1 q/seq


def gather_packed(kc, vc):
    """Current boolean-index gather (NOT graph-safe)."""
    gk = kc[block_table].reshape(B, M, Hkv, D)
    gv = vc[block_table].reshape(B, M, Hkv, D)
    pos = torch.arange(M, device=dev).unsqueeze(0)
    mask = pos < seqlens.long().unsqueeze(1)
    return gk[mask], gv[mask], mask  # [total_k, Hkv, D]


def scatter_packed(kc, vc):
    """Graph-safe: fixed-size scatter into a fixed buffer."""
    gk = kc[block_table].reshape(B, M, Hkv, D)
    gv = vc[block_table].reshape(B, M, Hkv, D)
    sl = seqlens.long()
    offsets = torch.zeros(B + 1, dtype=torch.long, device=dev)
    offsets[1:] = sl.cumsum(0)
    pos = torch.arange(M, device=dev).long()  # [M]
    dest = offsets[:-1, None] + pos[None, :]  # [B, M] = start_offset[b] + p
    valid = pos[None, :] < sl[:, None]  # [B, M]
    dest = torch.where(valid, dest, torch.full_like(dest, MAX_TOTAL - 1))  # invalid->dummy
    dest_flat = dest.reshape(B * M)  # [B*M]
    idx = dest_flat.view(-1, 1, 1).expand(-1, Hkv, D)
    pk = torch.zeros(MAX_TOTAL, Hkv, D, dtype=kc.dtype, device=dev)
    pv = torch.zeros(MAX_TOTAL, Hkv, D, dtype=kc.dtype, device=dev)
    pk.scatter_(0, idx, gk.reshape(B * M, Hkv, D))
    pv.scatter_(0, idx, gv.reshape(B * M, Hkv, D))
    return pk, pv, offsets


def sdpa_ref():
    ratio = Hq // Hkv
    refs = []
    for b in range(B):
        s = int(seqlens[b])
        kv = k_cache[block_table[b]].reshape(M, Hkv, D)[:s]
        vv = v_cache[block_table[b]].reshape(M, Hkv, D)[:s]
        qb4 = q[b].unsqueeze(1).unsqueeze(0)
        kv_e = kv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        vv_e = vv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        refs.append(
            F.scaled_dot_product_attention(qb4, kv_e, vv_e, is_causal=False, scale=scale)
            .squeeze(0).transpose(0, 1).reshape(1, Hq, D)
        )
    return torch.cat(refs, 0).float()


def main():
    total_k = int(seqlens.sum().item())
    gk_g, gv_g, _ = gather_packed(k_cache, v_cache)
    pk_s, pv_s, offsets = scatter_packed(k_cache, v_cache)

    # (a) scatter-packed valid portion == boolean-gather-packed
    mismatch = (pk_s[:total_k] - gk_g).abs().max().item()
    print(f"[a] scatter-packed vs gather-packed (first {total_k} valid): "
          f"max_err={mismatch:.4e}  {'OK' if mismatch == 0 else 'FAIL'}")

    # (b) flash_attn_varlen_func(scatter-packed) vs SDPA
    cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    cu_k = offsets.to(torch.int32)
    out = flash_attn_varlen_func(
        q, pk_s, pv_s, cu_q, cu_k,
        max_seqlen_q=1, max_seqlen_k=M,  # M = fixed upper bound (graph-safe)
        softmax_scale=scale, causal=False,
    )
    out = out.float()
    ref = sdpa_ref()
    err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
    print(f"[b] varlen(scatter-packed) vs SDPA: max_err={err:.4e}  "
          f"{'OK' if ok else 'FAIL'}  shape={tuple(out.shape)}")
    # note: no .item() on seqlens in the hot path -> graph-safe (cu_k is a tensor)


if __name__ == "__main__":
    main()
