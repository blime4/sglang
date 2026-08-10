"""DL: correctness smoke for the from-source _vllm_fa2_C.so (Route B).

Calls the exact clean decode path sglang uses —
`vllm_flash_attn.flash_attn_varlen_func(... block_table=...)` on paged KV —
and compares against torch SDPA on the gathered KV. Also saves the output so
two builds (from-source vs copied-from-vLLM) can be diffed bit-for-bit.

Run under each .so build:
    python scripts/dl/test_dlin_vllm_fa2_correctness.py built   # -> /tmp/fa2_out_built.pt
    python scripts/dl/test_dlin_vllm_fa2_correctness.py copied  # -> /tmp/fa2_out_copied.pt
"""
import sys
import torch
import torch.nn.functional as F

import vllm_flash_attn as vfa

label = sys.argv[1] if len(sys.argv) > 1 else "built"
torch.manual_seed(42)

# Qwen3-1.7B decode shape: GQA 16/8 heads, head_dim 128, page 16.
B, Hq, Hkv, D, Pg = 2, 16, 8, 128, 16
max_blocks = 4
device, dtype = "cuda", torch.bfloat16
rep = Hq // Hkv
scale = D ** -0.5

q = torch.randn(B, Hq, D, dtype=dtype, device=device)
k_cache = torch.randn(max_blocks, Pg, Hkv, D, dtype=dtype, device=device) * 0.3
v_cache = torch.randn(max_blocks, Pg, Hkv, D, dtype=dtype, device=device) * 0.3
block_table = torch.randint(0, max_blocks, (B, max_blocks), dtype=torch.int32, device=device)
cache_seqlens = torch.tensor([Pg * 2 + 3, Pg + 5], dtype=torch.int32, device=device)

cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device)
max_k_ub = max_blocks * Pg

out = vfa.flash_attn_varlen_func(
    q=q, k=k_cache, v=v_cache,
    max_seqlen_q=1, cu_seqlens_q=cu_q,
    max_seqlen_k=max_k_ub, seqused_k=cache_seqlens,
    softmax_scale=scale, causal=True, block_table=block_table,
)  # [B, Hq, D]
print(f"[{label}] vllm_flash_attn out: {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")

# Reference: gather paged KV per-seq, run SDPA (kv heads repeated for GQA).
out_ref = torch.empty_like(out)
for b in range(B):
    sl = int(cache_seqlens[b].item())
    gk = k_cache[block_table[b]].reshape(-1, Hkv, D)[:sl]  # [sl, Hkv, D]
    gv = v_cache[block_table[b]].reshape(-1, Hkv, D)[:sl]
    gk = gk.transpose(0, 1).unsqueeze(0).repeat_interleave(rep, dim=1)  # [1, Hq, sl, D]
    gv = gv.transpose(0, 1).unsqueeze(0).repeat_interleave(rep, dim=1)
    qb = q[b].unsqueeze(1).unsqueeze(0)  # [1, Hq, 1, D]
    o = F.scaled_dot_product_attention(qb, gk, gv, scale=scale)  # [1, Hq, 1, D]
    out_ref[b] = o.squeeze(0).squeeze(1)

diff = (out.float() - out_ref.float()).abs()
print(f"[{label}] vs SDPA: max_err={diff.max().item():.4f} "
      f"mean_err={diff.mean().item():.5f} out_absmax={out.abs().max().item():.4f}")
print(f"[{label}] checksum out={out.float().sum().item():.4f} "
      f"out_ref={out_ref.float().sum().item():.4f}")

out_path = f"/tmp/fa2_out_{label}.pt"
torch.save(out.cpu(), out_path)
print(f"[{label}] saved {out_path}")
