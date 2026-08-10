#!/usr/bin/env python3
# Validate the clean DLIN vllm_flash_attn decode path (varlen + block_table),
# the graph-safe path sglang uses when a compiled _vllm_fa2_C is available
# (i.e. the dl24 vllm_flash_attn wheel is installed). Compares vs torch SDPA.
# SKIPS cleanly if no compiled _vllm_fa2_C (vanilla stub) -- in that case sglang
# uses the gather workaround instead. See docs/dl/request-dl24-vllm-flash-attn-wheel.md.
import torch
import torch.nn.functional as F


def _have_dlin_vfa() -> bool:
    try:
        import vllm_flash_attn  # noqa: F401

        _ = torch.ops._vllm_fa2_C.varlen_fwd  # raises if no compiled _C
        return True
    except Exception:
        return False


def main():
    if not _have_dlin_vfa():
        print(
            "SKIP: no compiled DLIN vllm_flash_attn (_vllm_fa2_C.varlen_fwd absent). "
            "Install the dl24 wheel per docs/dl/request-dl24-vllm-flash-attn-wheel.md, "
            "then re-run. sglang currently falls back to the gather workaround."
        )
        return
    import vllm_flash_attn

    torch.manual_seed(0)
    dev = "cuda"
    B, Hq, Hkv, D, Pg = 4, 16, 8, 128, 16
    seqlens = torch.tensor([3, 1, 21, 35], dtype=torch.int32, device=dev)
    max_blocks = (int(seqlens.max().item()) + Pg - 1) // Pg
    num_blocks = 64
    scale = 1.0 / (D ** 0.5)
    kc = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
    vc = torch.randn(num_blocks, Pg, Hkv, D, dtype=torch.bfloat16, device=dev) * 0.3
    bt = torch.randint(0, num_blocks, (B, max_blocks), dtype=torch.int32, device=dev)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=dev) * 0.3  # [B, Hq, D], 1 q/seq
    cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    max_k_ub = max_blocks * Pg  # shape-derived upper bound (graph-safe)

    out = vllm_flash_attn.flash_attn_varlen_func(
        q=q, k=kc, v=vc, max_seqlen_q=1, cu_seqlens_q=cu_q, max_seqlen_k=max_k_ub,
        seqused_k=seqlens, softmax_scale=scale, causal=False, block_table=bt,
    )
    out = out.float()

    ratio = Hq // Hkv
    refs = []
    for b in range(B):
        s = int(seqlens[b])
        kv = kc[bt[b]].reshape(max_blocks * Pg, Hkv, D)[:s]
        vv = vc[bt[b]].reshape(max_blocks * Pg, Hkv, D)[:s]
        qb4 = q[b].unsqueeze(1).unsqueeze(0)
        kv_e = kv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        vv_e = vv.permute(1, 0, 2).repeat_interleave(ratio, dim=0).unsqueeze(0)
        refs.append(
            F.scaled_dot_product_attention(qb4, kv_e, vv_e, is_causal=False, scale=scale)
            .squeeze(0).transpose(0, 1).reshape(1, Hq, D)
        )
    ref = torch.cat(refs, 0).float()
    err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=5e-2, rtol=5e-2)
    print(f"vllm_flash_attn varlen+block_table vs SDPA: max_err={err:.4e}  "
          f"{'OK' if ok else 'FAIL'}  shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
