#!/usr/bin/env python3
"""Validate a vectorized torch fallback for reconstruct_indices_from_tree_mask
(the sgl_kernel op missing in the DLIN build) against a brute-force reference
that directly mirrors the CUDA kernel in sgl-kernel/csrc/speculative/ngram_utils.cu.

Runs the official test oracle (sgl-kernel/tests/speculative/test_ngram_utils.py)
plus randomized fuzzing. Exits non-zero on any mismatch.
"""
import torch

BIG = 1 << 30


def brute_ref(tree_mask, verified_seq_len, D, bs):
    """Direct per-element port of the CUDA kernel. Obviously correct."""
    tm = tree_mask.view(bs, D, D).bool()
    vsl = verified_seq_len.view(bs)
    retrive_index = torch.zeros(bs, D, dtype=torch.int64)
    positions = torch.zeros(bs, D, dtype=torch.int64)
    next_token = torch.full((bs, D), -1, dtype=torch.int64)
    next_sibling = torch.full((bs, D), -1, dtype=torch.int64)
    for b in range(bs):
        for tid in range(D):
            depth = 0
            parent_idx = -1
            for i in range(tid - 1, -1, -1):
                if tm[b, tid, i]:
                    depth += 1
                    if parent_idx == -1:
                        parent_idx = i
            positions[b, tid] = depth + vsl[b]
            retrive_index[b, tid] = b * D + tid
            # next_token
            nt = -1
            for i in range(tid + 1, D):
                if tm[b, i, tid]:
                    nt = i
                    break
            next_token[b, tid] = nt
            # next_sibling
            ns = -1
            if parent_idx != -1:
                for i in range(tid + 1, D):
                    if tm[b, i, parent_idx]:
                        is_sib = True
                        for j in range(parent_idx + 1, i):
                            if tm[b, i, j]:
                                is_sib = False
                                break
                        if is_sib:
                            ns = i
                            break
            next_sibling[b, tid] = ns
    return retrive_index, positions, next_token, next_sibling


def vec_impl(tree_mask, verified_seq_len, D, bs):
    """Vectorized torch fallback (target for DLIN)."""
    device = tree_mask.device
    tm = tree_mask.view(bs, D, D).bool()
    aD = torch.arange(D, device=device)
    row = aD.view(1, D, 1)   # tid (dim1)
    col = aD.view(1, 1, D)   # j  (dim2)

    lower = col < row                       # j < tid
    parent_mask = tm & lower                 # [bs,D,D]
    depth = parent_mask.sum(-1)              # [bs,D]
    pvals = torch.where(parent_mask, col.expand(bs, D, D),
                        torch.full((bs, D, D), -1, device=device, dtype=torch.int64))
    parent_idx = pvals.max(-1).values        # [bs,D]

    gidx = torch.arange(bs, device=device).view(bs, 1) * D + aD.view(1, D)
    pos2d = depth + verified_seq_len.view(bs, 1)

    # next_token: min i>tid with tm[b,i,tid] True
    tm_t = tm.transpose(1, 2)                # [bs,D,D], tm_t[b,tid,i]=tm[b,i,tid]
    upper = col > row                         # i > tid (dim1=tid, dim2=i)
    nt_mask = tm_t & upper
    nt_vals = torch.where(nt_mask, col.expand(bs, D, D),
                          torch.full((bs, D, D), BIG, device=device, dtype=torch.int64))
    next_token = torch.where(nt_vals.min(-1).values < BIG,
                             nt_vals.min(-1).values,
                             torch.full((bs, D), -1, device=device, dtype=torch.int64))

    # next_sibling: min i>tid with parent_idx[b,i]==parent_idx[b,tid], parent!=-1
    pi_t = parent_idx.view(bs, D, 1)
    pi_i = parent_idx.view(bs, 1, D)
    sib_mask = (pi_i == pi_t) & upper & (pi_t != -1)
    ns_vals = torch.where(sib_mask, col.expand(bs, D, D),
                          torch.full((bs, D, D), BIG, device=device, dtype=torch.int64))
    next_sibling = torch.where(ns_vals.min(-1).values < BIG,
                               ns_vals.min(-1).values,
                               torch.full((bs, D), -1, device=device, dtype=torch.int64))
    return gidx, pos2d, next_token, next_sibling


def run_one(tree_mask, vsl, D, bs, device):
    ri_b, po_b, nt_b, ns_b = brute_ref(tree_mask, vsl, D, bs)
    ri_v, po_v, nt_v, ns_v = vec_impl(tree_mask, vsl, D, bs)
    ok = (torch.equal(ri_b, ri_v) and torch.equal(po_b.cpu(), po_v.cpu())
          and torch.equal(nt_b, nt_v) and torch.equal(ns_b, ns_v))
    return ok, (ri_b, po_b, nt_b, ns_b), (ri_v, po_v, nt_v, ns_v)


def main():
    # --- Official oracle test case ---
    tm = torch.tensor([1,0,0,0, 1,1,0,0, 1,0,1,0, 1,0,1,1], dtype=torch.int32).bool().view(1,4,4)
    vsl = torch.tensor([12])
    ok, ref, vec = run_one(tm, vsl, 4, 1, "cpu")
    print(f"[oracle] match={ok}")
    assert ok, f"ORACLE MISMATCH\n ref={ref}\n vec={vec}"
    ri, po, nt, ns = vec
    assert ri.tolist() == [[0,1,2,3]], ri
    assert nt.tolist() == [[1,-1,3,-1]], nt
    assert ns.tolist() == [[-1,2,-1,-1]], ns
    assert po.tolist() == [[12,13,13,14]], po
    print("[oracle] exact expected values verified")

    # --- Randomized fuzzing ---
    torch.manual_seed(0)
    fails = 0
    trials = 3000
    for _ in range(trials):
        bs = torch.randint(1, 5, (1,)).item()
        D = torch.randint(1, 9, (1,)).item()
        # random DAG-ish mask (force self-True on diagonal for realism, but test logic is mask-agnostic)
        tm = (torch.rand(bs, D, D) < 0.5).bool()
        vsl = torch.randint(0, 100, (bs,))
        ok, _, _ = run_one(tm, vsl, D, bs, "cpu")
        if not ok:
            fails += 1
            if fails <= 3:
                print(f"  MISMATCH bs={bs} D={D}\n tm={tm.int()}\n vsl={vsl}")
    print(f"[fuzz] {trials} trials, {fails} mismatches")
    assert fails == 0, "FUZZ FAILURES"
    print("ALL PASS")


if __name__ == "__main__":
    main()
