from __future__ import annotations

import math
from functools import cache

import torch

import triton
import triton.language as tl

# Keep this file easy to audit against the original TileLang path:
# 1. Shared helpers.
# 2. Low-level Triton kernels mirroring vllm/_tilelang_ops.py order.
# 3. vLLM custom-op wrappers mirroring kernels/mhc/tilelang.py order.
# 4. OOT CustomOp replacements for layers/mhc.py.


# === shared helpers begin ===


def _check_tensor_dtype(
    tensor: torch.Tensor,
    expected: torch.dtype,
    *,
    tensor_name: str,
    fn_name: str,
) -> None:
    if tensor.dtype != expected:
        raise ValueError(
            f"{fn_name} expects {tensor_name} dtype {expected}, got {tensor.dtype}"
        )


def _prev_power_of_two(value: int) -> int:
    value = max(1, value)
    return 1 << (value.bit_length() - 1)


def _next_power_of_two(value: int) -> int:
    value = max(1, value)
    return 1 << (value - 1).bit_length()


def _call_tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: int | None,
) -> None:
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)


# === shared helpers end ===


# Low-level Triton kernels mirroring vllm/_tilelang_ops.py.
# Public function order intentionally matches:
# compute_num_split -> mhc_pre_big_fuse -> mhc_fused -> mhc_post -> hc_head_fuse.


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        num_block_k = math.ceil(k / block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


# === mhc_pre_big_fuse begin ===


@triton.jit
def _mhc_pre_big_fuse_triton_kernel(
    gemm_out_mul_ptr,
    gemm_out_sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    num_tokens,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits,
    hc_mult,
    BLOCK_SPLIT: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_COMB: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_SINKHORN: tl.constexpr,
    USE_EXACT_SINKHORN: tl.constexpr,
    USE_FULL_CORE_TILE: tl.constexpr,
    PARALLEL_H_TILES: tl.constexpr,
    USE_MASKLESS_RESIDUAL_LOOP: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_hc = tl.arange(0, BLOCK_HC)
    offs_comb = tl.arange(0, BLOCK_COMB)
    mask_hc = offs_hc < hc_mult
    mask_comb = offs_comb < (hc_mult * hc_mult)

    hc_mult3 = hc_mult * (2 + hc_mult)
    token_offset = pid_n * hc_mult3

    sqrsum = tl.zeros((), dtype=tl.float32)
    pre_sums = tl.zeros((BLOCK_HC,), dtype=tl.float32)
    post_sums = tl.zeros((BLOCK_HC,), dtype=tl.float32)
    comb_sums = tl.zeros((BLOCK_COMB,), dtype=tl.float32)

    if USE_FULL_CORE_TILE:
        if PARALLEL_H_TILES:
            for i_split in tl.static_range(BLOCK_SPLIT):
                sqrsum += tl.load(
                    gemm_out_sqrsum_ptr + i_split * num_tokens + pid_n,
                ).to(tl.float32)

                split_base = i_split * num_tokens * hc_mult3 + token_offset
                pre_sums += tl.load(gemm_out_mul_ptr + split_base + offs_hc).to(
                    tl.float32
                )

            if pid_h == 0:
                for i_split in tl.static_range(BLOCK_SPLIT):
                    split_base = i_split * num_tokens * hc_mult3 + token_offset
                    post_sums += tl.load(
                        gemm_out_mul_ptr + split_base + hc_mult + offs_hc,
                    ).to(tl.float32)
                    comb_sums += tl.load(
                        gemm_out_mul_ptr + split_base + 2 * hc_mult + offs_comb,
                    ).to(tl.float32)
        else:
            for i_split in tl.static_range(BLOCK_SPLIT):
                sqrsum += tl.load(
                    gemm_out_sqrsum_ptr + i_split * num_tokens + pid_n,
                ).to(tl.float32)

                split_base = i_split * num_tokens * hc_mult3 + token_offset
                pre_sums += tl.load(gemm_out_mul_ptr + split_base + offs_hc).to(
                    tl.float32
                )
                post_sums += tl.load(
                    gemm_out_mul_ptr + split_base + hc_mult + offs_hc,
                ).to(tl.float32)
                comb_sums += tl.load(
                    gemm_out_mul_ptr + split_base + 2 * hc_mult + offs_comb,
                ).to(tl.float32)
    else:
        for i_split in tl.static_range(BLOCK_SPLIT):
            active_split = i_split < n_splits
            sqrsum += tl.load(
                gemm_out_sqrsum_ptr + i_split * num_tokens + pid_n,
                mask=active_split,
                other=0.0,
            ).to(tl.float32)

            split_base = i_split * num_tokens * hc_mult3 + token_offset
            pre_sums += tl.load(
                gemm_out_mul_ptr + split_base + offs_hc,
                mask=mask_hc & active_split,
                other=0.0,
            ).to(tl.float32)
            post_sums += tl.load(
                gemm_out_mul_ptr + split_base + hc_mult + offs_hc,
                mask=mask_hc & active_split,
                other=0.0,
            ).to(tl.float32)
            comb_sums += tl.load(
                gemm_out_mul_ptr + split_base + 2 * hc_mult + offs_comb,
                mask=mask_comb & active_split,
                other=0.0,
            ).to(tl.float32)

    rms = tl.rsqrt(sqrsum / tl.cast(hc_mult * hidden_size, tl.float32) + rms_eps)

    hc_scale_pre = tl.load(hc_scale_ptr + 0).to(tl.float32)
    hc_scale_post = tl.load(hc_scale_ptr + 1).to(tl.float32)
    hc_scale_comb = tl.load(hc_scale_ptr + 2).to(tl.float32)

    if USE_FULL_CORE_TILE:
        pre_logits = pre_sums * rms * hc_scale_pre + tl.load(
            hc_base_ptr + offs_hc,
        ).to(tl.float32)
        pre_mix = tl.sigmoid(pre_logits) + hc_pre_eps
        if PARALLEL_H_TILES:
            if pid_h == 0:
                post_logits = post_sums * rms * hc_scale_post + tl.load(
                    hc_base_ptr + hc_mult + offs_hc,
                ).to(tl.float32)
                post_vals = tl.sigmoid(post_logits) * hc_post_mult_value
                tl.store(post_mix_ptr + pid_n * hc_mult + offs_hc, post_vals)

                comb = hc_scale_comb * tl.reshape(
                    comb_sums * rms, (BLOCK_HC, BLOCK_HC)
                ) + tl.load(
                    hc_base_ptr + 2 * hc_mult + offs_comb,
                ).to(tl.float32).reshape((BLOCK_HC, BLOCK_HC))

                row_max = tl.max(comb, axis=1)
                comb = tl.exp(comb - row_max[:, None])
                row_sum = tl.sum(comb, axis=1)
                comb = comb / row_sum[:, None] + hc_sinkhorn_eps

                col_sum = tl.sum(comb, axis=0)
                comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)

                for iter_idx in range(sinkhorn_repeat - 1):
                    row_sum = tl.sum(comb, axis=1)
                    next_comb = comb / (row_sum[:, None] + hc_sinkhorn_eps)
                    if USE_EXACT_SINKHORN:
                        comb = next_comb
                    else:
                        active_iter = iter_idx < (sinkhorn_repeat - 1)
                        comb = tl.where(active_iter, next_comb, comb)
                    col_sum = tl.sum(comb, axis=0)
                    next_comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)
                    if USE_EXACT_SINKHORN:
                        comb = next_comb
                    else:
                        comb = tl.where(active_iter, next_comb, comb)

                tl.store(
                    comb_mix_ptr + pid_n * hc_mult * hc_mult + offs_comb,
                    tl.reshape(comb, (BLOCK_COMB,)),
                )
        else:
            post_logits = post_sums * rms * hc_scale_post + tl.load(
                hc_base_ptr + hc_mult + offs_hc,
            ).to(tl.float32)
            post_vals = tl.sigmoid(post_logits) * hc_post_mult_value
            tl.store(post_mix_ptr + pid_n * hc_mult + offs_hc, post_vals)

            comb = hc_scale_comb * tl.reshape(
                comb_sums * rms, (BLOCK_HC, BLOCK_HC)
            ) + tl.load(
                hc_base_ptr + 2 * hc_mult + offs_comb,
            ).to(tl.float32).reshape((BLOCK_HC, BLOCK_HC))

            row_max = tl.max(comb, axis=1)
            comb = tl.exp(comb - row_max[:, None])
            row_sum = tl.sum(comb, axis=1)
            comb = comb / row_sum[:, None] + hc_sinkhorn_eps

            col_sum = tl.sum(comb, axis=0)
            comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)

            for iter_idx in range(sinkhorn_repeat - 1):
                row_sum = tl.sum(comb, axis=1)
                next_comb = comb / (row_sum[:, None] + hc_sinkhorn_eps)
                if USE_EXACT_SINKHORN:
                    comb = next_comb
                else:
                    active_iter = iter_idx < (sinkhorn_repeat - 1)
                    comb = tl.where(active_iter, next_comb, comb)
                col_sum = tl.sum(comb, axis=0)
                next_comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)
                if USE_EXACT_SINKHORN:
                    comb = next_comb
                else:
                    comb = tl.where(active_iter, next_comb, comb)

            tl.store(
                comb_mix_ptr + pid_n * hc_mult * hc_mult + offs_comb,
                tl.reshape(comb, (BLOCK_COMB,)),
            )
    else:
        pre_logits = (
            pre_sums * rms * hc_scale_pre
            + tl.load(hc_base_ptr + offs_hc, mask=mask_hc, other=0.0).to(
                tl.float32
            )
        )
        pre_mix = tl.where(mask_hc, tl.sigmoid(pre_logits) + hc_pre_eps, 0.0)
        post_logits = (
            post_sums * rms * hc_scale_post
            + tl.load(hc_base_ptr + hc_mult + offs_hc, mask=mask_hc, other=0.0).to(
                tl.float32
            )
        )
        post_vals = tl.where(
            mask_hc, tl.sigmoid(post_logits) * hc_post_mult_value, 0.0
        )
        tl.store(
            post_mix_ptr + pid_n * hc_mult + offs_hc,
            post_vals,
            mask=mask_hc,
        )

        comb = hc_scale_comb * tl.reshape(
            comb_sums * rms, (BLOCK_HC, BLOCK_HC)
        ) + tl.load(
            hc_base_ptr + 2 * hc_mult + offs_comb,
            mask=mask_comb,
            other=0.0,
        ).to(tl.float32).reshape((BLOCK_HC, BLOCK_HC))
        row_idx = tl.arange(0, BLOCK_HC)[:, None]
        col_idx = tl.arange(0, BLOCK_HC)[None, :]
        valid_comb = (row_idx < hc_mult) & (col_idx < hc_mult)

        comb = tl.where(valid_comb, comb, -float("inf"))
        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        comb = tl.where(valid_comb, comb, 0.0)
        row_sum = tl.sum(comb, axis=1)
        comb = tl.where(valid_comb, comb / row_sum[:, None] + hc_sinkhorn_eps, 0.0)

        col_sum = tl.sum(comb, axis=0)
        comb = tl.where(
            valid_comb, comb / (col_sum[None, :] + hc_sinkhorn_eps), 0.0
        )

        for iter_idx in range(sinkhorn_repeat - 1):
            row_sum = tl.sum(comb, axis=1)
            next_comb = tl.where(
                valid_comb, comb / (row_sum[:, None] + hc_sinkhorn_eps), 0.0
            )
            if USE_EXACT_SINKHORN:
                comb = next_comb
            else:
                active_iter = iter_idx < (sinkhorn_repeat - 1)
                comb = tl.where(active_iter, next_comb, comb)
            col_sum = tl.sum(comb, axis=0)
            next_comb = tl.where(
                valid_comb, comb / (col_sum[None, :] + hc_sinkhorn_eps), 0.0
            )
            if USE_EXACT_SINKHORN:
                comb = next_comb
            else:
                comb = tl.where(active_iter, next_comb, comb)

        tl.store(
            comb_mix_ptr + pid_n * hc_mult * hc_mult + offs_comb,
            tl.reshape(comb, (BLOCK_COMB,)),
            mask=mask_comb,
        )

    residual_base = residual_ptr + pid_n * hc_mult * hidden_size
    if USE_MASKLESS_RESIDUAL_LOOP:
        # Same residual -> layer_input loop as the fallback path, with masks
        # removed. The wrapper only enables it for full hidden and HC tiles.
        if PARALLEL_H_TILES:
            if pid_h != 0:
                offs_h = (pid_h - 1) * BLOCK_H + tl.arange(0, BLOCK_H)
                out_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
                for i_hc in tl.static_range(BLOCK_HC):
                    residual_row = tl.load(
                        residual_base + i_hc * hidden_size + offs_h,
                    ).to(tl.float32)
                    pre_scalar = tl.sum(
                        tl.where(offs_hc == i_hc, pre_mix, 0.0), axis=0
                    )
                    out_acc += pre_scalar * residual_row
                tl.store(layer_input_ptr + pid_n * hidden_size + offs_h, out_acc)
        else:
            for i_h in tl.static_range(NUM_H_BLOCKS):
                offs_h = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
                out_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
                for i_hc in tl.static_range(BLOCK_HC):
                    residual_row = tl.load(
                        residual_base + i_hc * hidden_size + offs_h,
                    ).to(tl.float32)
                    pre_scalar = tl.sum(
                        tl.where(offs_hc == i_hc, pre_mix, 0.0), axis=0
                    )
                    out_acc += pre_scalar * residual_row
                tl.store(layer_input_ptr + pid_n * hidden_size + offs_h, out_acc)
    else:
        for i_h in tl.static_range(NUM_H_BLOCKS):
            offs_h = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < hidden_size
            out_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for i_hc in tl.static_range(BLOCK_HC):
                active_hc = i_hc < hc_mult
                residual_row = tl.load(
                    residual_base + i_hc * hidden_size + offs_h,
                    mask=mask_h & active_hc,
                    other=0.0,
                ).to(tl.float32)
                pre_scalar = tl.sum(tl.where(offs_hc == i_hc, pre_mix, 0.0), axis=0)
                out_acc += pre_scalar * residual_row
            tl.store(
                layer_input_ptr + pid_n * hidden_size + offs_h,
                out_acc,
                mask=mask_h,
            )


@triton.jit
def _mhc_pre_big_fuse_triton_single_token_hc4_kernel(
    gemm_out_mul_ptr,
    gemm_out_sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_SINKHORN: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)

    offs_hc = tl.arange(0, 4)
    offs_comb = tl.arange(0, 16)

    sqrsum = tl.zeros((), dtype=tl.float32)
    pre_sums = tl.zeros((4,), dtype=tl.float32)
    post_sums = tl.zeros((4,), dtype=tl.float32)
    comb_sums = tl.zeros((16,), dtype=tl.float32)

    for i_split in tl.static_range(BLOCK_SPLIT):
        sqrsum += tl.load(gemm_out_sqrsum_ptr + i_split).to(tl.float32)

        split_base = i_split * 24
        pre_sums += tl.load(gemm_out_mul_ptr + split_base + offs_hc).to(
            tl.float32
        )

    if pid_h == 0:
        for i_split in tl.static_range(BLOCK_SPLIT):
            split_base = i_split * 24
            post_sums += tl.load(
                gemm_out_mul_ptr + split_base + 4 + offs_hc,
            ).to(tl.float32)
            comb_sums += tl.load(
                gemm_out_mul_ptr + split_base + 8 + offs_comb,
            ).to(tl.float32)

    rms = tl.rsqrt(sqrsum / (4.0 * HIDDEN_SIZE) + rms_eps)

    hc_scale_pre = tl.load(hc_scale_ptr + 0).to(tl.float32)
    hc_scale_post = tl.load(hc_scale_ptr + 1).to(tl.float32)
    hc_scale_comb = tl.load(hc_scale_ptr + 2).to(tl.float32)

    pre_logits = pre_sums * rms * hc_scale_pre + tl.load(
        hc_base_ptr + offs_hc,
    ).to(tl.float32)
    pre_mix = tl.sigmoid(pre_logits) + hc_pre_eps

    if pid_h == 0:
        post_logits = post_sums * rms * hc_scale_post + tl.load(
            hc_base_ptr + 4 + offs_hc,
        ).to(tl.float32)
        post_vals = tl.sigmoid(post_logits) * hc_post_mult_value
        tl.store(post_mix_ptr + offs_hc, post_vals)

        comb = hc_scale_comb * tl.reshape(
            comb_sums * rms, (4, 4)
        ) + tl.load(
            hc_base_ptr + 8 + offs_comb,
        ).to(tl.float32).reshape((4, 4))

        row_max = tl.max(comb, axis=1)
        comb = tl.exp(comb - row_max[:, None])
        row_sum = tl.sum(comb, axis=1)
        comb = comb / row_sum[:, None] + hc_sinkhorn_eps

        col_sum = tl.sum(comb, axis=0)
        comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)

        for _ in range(sinkhorn_repeat - 1):
            row_sum = tl.sum(comb, axis=1)
            comb = comb / (row_sum[:, None] + hc_sinkhorn_eps)
            col_sum = tl.sum(comb, axis=0)
            comb = comb / (col_sum[None, :] + hc_sinkhorn_eps)

        tl.store(comb_mix_ptr + offs_comb, tl.reshape(comb, (16,)))

    if pid_h != 0:
        offs_h = (pid_h - 1) * BLOCK_H + tl.arange(0, BLOCK_H)
        out_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i_hc in tl.static_range(4):
            residual_row = tl.load(
                residual_ptr + i_hc * HIDDEN_SIZE + offs_h,
            ).to(tl.float32)
            pre_scalar = tl.sum(tl.where(offs_hc == i_hc, pre_mix, 0.0), axis=0)
            out_acc += pre_scalar * residual_row
        tl.store(layer_input_ptr + offs_h, out_acc)


def mhc_pre_big_fuse_triton(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mhc_pre_big_fuse_triton")
    if hc_mult <= 0 or hidden_size <= 0:
        return

    if gemm_out_mul.ndim != 3 or gemm_out_sqrsum.ndim != 2 or residual.ndim != 3:
        raise ValueError(
            "mhc_pre_big_fuse_triton expects gemm_out_mul rank-3, "
            "gemm_out_sqrsum rank-2, residual rank-3"
        )
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_out_mul.shape != (n_splits, residual.shape[0], hc_mult3):
        raise ValueError(
            "Expected gemm_out_mul shape "
            f"({n_splits},{residual.shape[0]},{hc_mult3}), "
            f"got {tuple(gemm_out_mul.shape)}"
        )
    if gemm_out_sqrsum.shape != (n_splits, residual.shape[0]):
        raise ValueError(
            "Expected gemm_out_sqrsum shape "
            f"({n_splits},{residual.shape[0]}), "
            f"got {tuple(gemm_out_sqrsum.shape)}"
        )
    if hc_scale.shape != (3,):
        raise ValueError(f"Expected hc_scale shape (3,), got {tuple(hc_scale.shape)}")
    if hc_base.shape != (hc_mult3,):
        raise ValueError(
            f"Expected hc_base shape ({hc_mult3},), got {tuple(hc_base.shape)}"
        )
    if residual.shape != (residual.shape[0], hc_mult, hidden_size):
        raise ValueError(
            f"Expected residual shape (n,{hc_mult},{hidden_size}), "
            f"got {tuple(residual.shape)}"
        )
    if post_mix.shape != (residual.shape[0], hc_mult):
        raise ValueError(
            f"Expected post_mix shape (n,{hc_mult}), got {tuple(post_mix.shape)}"
        )
    if comb_mix.shape != (residual.shape[0], hc_mult * hc_mult):
        raise ValueError(
            f"Expected comb_mix shape (n,{hc_mult * hc_mult}), "
            f"got {tuple(comb_mix.shape)}"
        )
    if layer_input.shape != (residual.shape[0], hidden_size):
        raise ValueError(
            f"Expected layer_input shape (n,{hidden_size}), "
            f"got {tuple(layer_input.shape)}"
        )

    _check_tensor_dtype(
        gemm_out_mul,
        torch.float32,
        tensor_name="gemm_out_mul",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        gemm_out_sqrsum,
        torch.float32,
        tensor_name="gemm_out_sqrsum",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        hc_scale,
        torch.float32,
        tensor_name="hc_scale",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        hc_base,
        torch.float32,
        tensor_name="hc_base",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        residual,
        torch.bfloat16,
        tensor_name="residual",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        post_mix,
        torch.float32,
        tensor_name="post_mix",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        comb_mix,
        torch.float32,
        tensor_name="comb_mix",
        fn_name="mhc_pre_big_fuse_triton",
    )
    _check_tensor_dtype(
        layer_input,
        torch.bfloat16,
        tensor_name="layer_input",
        fn_name="mhc_pre_big_fuse_triton",
    )

    num_tokens = residual.shape[0]
    block_hc = _next_power_of_two(hc_mult)
    block_split = _next_power_of_two(n_splits)
    block_comb = block_hc * block_hc
    block_sinkhorn = max(1, sinkhorn_repeat - 1)
    use_exact_sinkhorn = sinkhorn_repeat > 1
    # The maskless full-core path is profitable for the anchor/lower-split
    # cases and tiny decode, but DL codegen becomes VR/LSU heavy for large
    # split counts with many tokens.
    prefer_full_core_tile = n_splits <= 8 or num_tokens <= 4
    use_full_core_tile = (
        n_splits == block_split and hc_mult == block_hc and prefer_full_core_tile
    )
    if hc_mult == 4 and hidden_size >= 1024:
        block_h = min(1024, hidden_size)
    else:
        block_h = math.gcd(512, hidden_size)
    num_h_blocks = triton.cdiv(hidden_size, block_h)

    # Maskless residual loads/stores require complete hidden and HC tiles.
    # Small decode and lower-split prefill use the maskless path; other ranges
    # stay masked.
    full_residual_tile = hidden_size % block_h == 0 and hc_mult == block_hc
    maskless_loop_preferred = num_tokens <= 24 or n_splits < 8
    use_maskless_residual_loop = full_residual_tile and maskless_loop_preferred
    parallel_h_tiles = (
        num_tokens == 1
        and use_full_core_tile
        and use_maskless_residual_loop
        and num_h_blocks > 1
    )

    # The tiny-token parallel path benefits from a lighter CTA; keep the
    # historical 4-warp shape for all other cases.
    num_warps = 2 if parallel_h_tiles else 4

    single_token_hc4_small_split = (
        num_tokens == 1
        and hc_mult == 4
        and n_splits <= 8
        and n_splits == block_split
        and sinkhorn_repeat > 1
        and hidden_size % block_h == 0
        and num_h_blocks > 1
    )
    if single_token_hc4_small_split:
        _mhc_pre_big_fuse_triton_single_token_hc4_kernel[
            (1, num_h_blocks + 1)
        ](
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SPLIT=block_split,
            BLOCK_H=block_h,
            NUM_H_BLOCKS=num_h_blocks,
            BLOCK_SINKHORN=block_sinkhorn,
            num_warps=2,
        )
        return

    grid = (num_tokens, num_h_blocks + 1) if parallel_h_tiles else (num_tokens,)
    _mhc_pre_big_fuse_triton_kernel[grid](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        num_tokens,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        BLOCK_SPLIT=block_split,
        BLOCK_HC=block_hc,
        BLOCK_COMB=block_comb,
        BLOCK_H=block_h,
        NUM_H_BLOCKS=num_h_blocks,
        BLOCK_SINKHORN=block_sinkhorn,
        USE_EXACT_SINKHORN=use_exact_sinkhorn,
        USE_FULL_CORE_TILE=use_full_core_tile,
        PARALLEL_H_TILES=parallel_h_tiles,
        USE_MASKLESS_RESIDUAL_LOOP=use_maskless_residual_loop,
        num_warps=num_warps,
    )


# === mhc_pre_big_fuse end ===


# === mhc_fused begin ===

def _select_mhc_fused_kernel_meta(
    hc: int,
    hidden: int,
    h_blk: int,
    tile_n: int,
    split_k: int,
    launch_programs: int,
) -> tuple[int, int, int, int, int, bool]:
    block_hc = _next_power_of_two(hc)
    hidden_per_split = hidden // split_k
    block_h = min(_next_power_of_two(hidden_per_split), h_blk)
    num_h_blocks = triton.cdiv(hidden_per_split, block_h)
    full_h_blocks = hidden_per_split % block_h == 0
    block_tile = _next_power_of_two(tile_n)
    # Small grids keep two warps for occupancy; larger grids use one warp to
    # reduce per-program work.
    wide_hidden_tile = block_h > 128
    many_programs = launch_programs >= 192
    use_one_warp = wide_hidden_tile and many_programs
    num_warps = 1 if use_one_warp else 2
    return block_hc, block_h, num_h_blocks, block_tile, num_warps, full_h_blocks


@triton.jit
def _mhc_fused_triton_kernel(
    comb_mix_ptr,
    residual_in_ptr,
    post_mix_ptr,
    x_in_ptr,
    weight_t_ptr,
    yp_out_ptr,
    rp_out_ptr,
    residual_out_ptr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    h_per_split: tl.constexpr,
    n_out: tl.constexpr,
    tile_n: tl.constexpr,
    split_k: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    FULL_H_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_nt = tl.program_id(axis=1)
    pid_ks = tl.program_id(axis=2)

    offs_tile = tl.arange(0, BLOCK_TILE)

    post_base = post_mix_ptr + pid_n * hc
    comb_base = comb_mix_ptr + pid_n * hc * hc
    residual_base = residual_in_ptr + pid_n * hc * hidden
    x_base = x_in_ptr + pid_n * hidden
    residual_out_base = residual_out_ptr + pid_n * hc * hidden

    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    acc3 = tl.zeros((), dtype=tl.float32)
    sqr = tl.zeros((), dtype=tl.float32)
    out_indices = pid_nt * tile_n + offs_tile
    mask_tile = offs_tile < tile_n
    out_idx0 = pid_nt * tile_n
    weight_base0 = weight_t_ptr + out_idx0 * hc * hidden
    if tile_n > 1:
        out_idx1 = out_idx0 + 1
        weight_base1 = weight_t_ptr + out_idx1 * hc * hidden
    if tile_n > 2:
        out_idx2 = out_idx0 + 2
        weight_base2 = weight_t_ptr + out_idx2 * hc * hidden
    if tile_n > 3:
        out_idx3 = out_idx0 + 3
        weight_base3 = weight_t_ptr + out_idx3 * hc * hidden

    split_h_start = pid_ks * h_per_split

    # Match the TileLang fused kernel at the same granularity: one program owns
    # one (token, n-tile, k-split) slice and fuses post-mapping, split-K GEMM
    # accumulation, and the sqr-sum reduction needed by the next pre block.
    for i_h in tl.static_range(NUM_H_BLOCKS):
        offs_h_local = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
        if FULL_H_BLOCKS:
            mask_h = tl.full((BLOCK_H,), True, dtype=tl.int1)
        else:
            mask_h = offs_h_local < h_per_split
        offs_h = split_h_start + offs_h_local

        x_vals = tl.load(
            x_base + offs_h,
            mask=mask_h,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if BLOCK_HC == 4:
            active_r0 = 0 < hc
            active_r1 = 1 < hc
            active_r2 = 2 < hc
            active_r3 = 3 < hc
            residual_row0 = tl.load(
                residual_base + 0 * hidden + offs_h,
                mask=mask_h & active_r0,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            residual_row1 = tl.load(
                residual_base + 1 * hidden + offs_h,
                mask=mask_h & active_r1,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            residual_row2 = tl.load(
                residual_base + 2 * hidden + offs_h,
                mask=mask_h & active_r2,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            residual_row3 = tl.load(
                residual_base + 3 * hidden + offs_h,
                mask=mask_h & active_r3,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
        for j in tl.static_range(BLOCK_HC):
            active_j = j < hc
            post_j = tl.load(post_base + j, mask=active_j, other=0.0).to(tl.float32)
            new_r_j = post_j * x_vals

            for k in tl.static_range(BLOCK_HC):
                active_k = k < hc
                if BLOCK_HC == 4:
                    if k == 0:
                        residual_row = residual_row0
                    elif k == 1:
                        residual_row = residual_row1
                    elif k == 2:
                        residual_row = residual_row2
                    else:
                        residual_row = residual_row3
                else:
                    residual_row = tl.load(
                        residual_base + k * hidden + offs_h,
                        mask=mask_h & active_k & active_j,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                coeff = tl.load(
                    comb_base + k * hc + j,
                    mask=active_k & active_j,
                    other=0.0,
                ).to(tl.float32)
                new_r_j += coeff * residual_row

            if pid_nt == 0:
                tl.store(
                    residual_out_base + j * hidden + offs_h,
                    new_r_j,
                    mask=mask_h & active_j,
                )
                sqr += tl.sum(new_r_j * new_r_j, axis=0)

            weight_row0 = tl.load(
                weight_base0 + j * hidden + offs_h,
                mask=mask_h & active_j,
                other=0.0,
            ).to(tl.float32)
            acc0 += tl.sum(weight_row0 * new_r_j, axis=0)

            if tile_n > 1:
                weight_row1 = tl.load(
                    weight_base1 + j * hidden + offs_h,
                    mask=mask_h & active_j,
                    other=0.0,
                ).to(tl.float32)
                acc1 += tl.sum(weight_row1 * new_r_j, axis=0)

            if tile_n > 2:
                weight_row2 = tl.load(
                    weight_base2 + j * hidden + offs_h,
                    mask=mask_h & active_j,
                    other=0.0,
                ).to(tl.float32)
                acc2 += tl.sum(weight_row2 * new_r_j, axis=0)

            if tile_n > 3:
                weight_row3 = tl.load(
                    weight_base3 + j * hidden + offs_h,
                    mask=mask_h & active_j,
                    other=0.0,
                ).to(tl.float32)
                acc3 += tl.sum(weight_row3 * new_r_j, axis=0)

    yp_base = (
        yp_out_ptr + pid_ks * (tl.num_programs(axis=0) * n_out) + pid_n * n_out
    )
    acc = tl.zeros((BLOCK_TILE,), dtype=tl.float32)
    acc = tl.where(offs_tile == 0, acc0, acc)
    if tile_n > 1:
        acc = tl.where(offs_tile == 1, acc1, acc)
    if tile_n > 2:
        acc = tl.where(offs_tile == 2, acc2, acc)
    if tile_n > 3:
        acc = tl.where(offs_tile == 3, acc3, acc)
    tl.store(yp_base + out_indices, acc, mask=mask_tile)

    if pid_nt == 0:
        rp_base = rp_out_ptr + pid_ks * tl.num_programs(axis=0) + pid_n
        tl.store(rp_base, sqr)


def mhc_fused_triton(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mhc_fused_triton")
    if hc <= 0 or hidden <= 0 or n_out <= 0:
        return

    num_tokens = comb_mix.shape[0]
    hidden_per_split = hidden // split_k
    n_tiles = n_out // tile_n

    if comb_mix.shape != (num_tokens, hc, hc):
        raise ValueError(
            f"Expected comb_mix shape (n,{hc},{hc}), got {tuple(comb_mix.shape)}"
        )
    if residual_in.shape != (num_tokens, hc, hidden):
        raise ValueError(
            f"Expected residual_in shape (n,{hc},{hidden}), "
            f"got {tuple(residual_in.shape)}"
        )
    if post_mix.shape != (num_tokens, hc):
        raise ValueError(
            f"Expected post_mix shape (n,{hc}), got {tuple(post_mix.shape)}"
        )
    if x_in.shape != (num_tokens, hidden):
        raise ValueError(f"Expected x_in shape (n,{hidden}), got {tuple(x_in.shape)}")
    if weight_t.shape != (n_out, hc, hidden):
        raise ValueError(
            f"Expected weight_t shape ({n_out},{hc},{hidden}), "
            f"got {tuple(weight_t.shape)}"
        )
    if yp_out.shape != (split_k, num_tokens, n_out):
        raise ValueError(
            f"Expected yp_out shape ({split_k},{num_tokens},{n_out}), "
            f"got {tuple(yp_out.shape)}"
        )
    if rp_out.shape != (split_k, num_tokens):
        raise ValueError(
            f"Expected rp_out shape ({split_k},{num_tokens}), got {tuple(rp_out.shape)}"
        )
    if residual_out.shape != (num_tokens, hc, hidden):
        raise ValueError(
            f"Expected residual_out shape (n,{hc},{hidden}), "
            f"got {tuple(residual_out.shape)}"
        )
    if hidden % split_k != 0:
        raise ValueError(f"Expected hidden % split_k == 0, got {hidden} % {split_k}")
    if n_out % tile_n != 0:
        raise ValueError(f"Expected n_out % tile_n == 0, got {n_out} % {tile_n}")

    _check_tensor_dtype(
        comb_mix,
        torch.float32,
        tensor_name="comb_mix",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        residual_in,
        torch.bfloat16,
        tensor_name="residual_in",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        post_mix,
        torch.float32,
        tensor_name="post_mix",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        x_in,
        torch.bfloat16,
        tensor_name="x_in",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        weight_t,
        torch.float32,
        tensor_name="weight_t",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        yp_out,
        torch.float32,
        tensor_name="yp_out",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        rp_out,
        torch.float32,
        tensor_name="rp_out",
        fn_name="mhc_fused_triton",
    )
    _check_tensor_dtype(
        residual_out,
        torch.bfloat16,
        tensor_name="residual_out",
        fn_name="mhc_fused_triton",
    )

    # n_thr remains in the public signature for wrapper/API parity. This kernel
    # selects warp count from the launch shape.
    launch_programs = num_tokens * n_tiles * split_k
    grid = (num_tokens, n_tiles, split_k)
    block_hc, block_h, num_h_blocks, block_tile, num_warps, full_h_blocks = (
        _select_mhc_fused_kernel_meta(
            hc,
            hidden,
            h_blk,
            tile_n,
            split_k,
            launch_programs,
        )
    )
    _mhc_fused_triton_kernel[grid](
        comb_mix,
        residual_in,
        post_mix,
        x_in,
        weight_t,
        yp_out,
        rp_out,
        residual_out,
        hc,
        hidden,
        hidden_per_split,
        n_out,
        tile_n,
        split_k,
        BLOCK_HC=block_hc,
        BLOCK_H=block_h,
        NUM_H_BLOCKS=num_h_blocks,
        BLOCK_TILE=block_tile,
        FULL_H_BLOCKS=full_h_blocks,
        num_warps=num_warps,
    )


# === mhc_fused end ===


# === mhc_post begin ===


@triton.jit
def _mhc_post_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    x_ptr,
    hc,
    hidden,
    BLOCK_HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_hco = tl.arange(0, BLOCK_HC)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_hco = offs_hco < hc
    mask_h = offs_h < hidden

    a_base = a_ptr + pid_n * hc * hc
    b_base = b_ptr + pid_n * hc * hidden
    c_base = c_ptr + pid_n * hc
    d_base = d_ptr + pid_n * hidden
    x_base = x_ptr + pid_n * hc * hidden

    c_vec = tl.load(c_base + offs_hco, mask=mask_hco, other=0.0).to(tl.float32)
    d_vec = tl.load(d_base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    acc = c_vec[:, None] * d_vec[None, :]

    for k in tl.static_range(BLOCK_HC):
        active_hci = k < hc
        a_row = tl.load(
            a_base + k * hc + offs_hco,
            mask=mask_hco & active_hci,
            other=0.0,
        ).to(tl.float32)
        b_row = tl.load(
            b_base + k * hidden + offs_h,
            mask=mask_h & active_hci,
            other=0.0,
        ).to(tl.float32)
        acc += a_row[:, None] * b_row[None, :]

    out_ptrs = x_base + offs_hco[:, None] * hidden + offs_h[None, :]
    out_mask = mask_hco[:, None] & mask_h[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _mhc_post_full_tile_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    x_ptr,
    hc,
    hidden,
    BLOCK_HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Caller guarantees complete HC and hidden tiles, so loads/stores can omit
    # masks without changing boundary behavior.
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_hco = tl.arange(0, BLOCK_HC)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    a_base = a_ptr + pid_n * hc * hc
    b_base = b_ptr + pid_n * hc * hidden
    c_base = c_ptr + pid_n * hc
    d_base = d_ptr + pid_n * hidden
    x_base = x_ptr + pid_n * hc * hidden

    c_vec = tl.load(c_base + offs_hco).to(tl.float32)
    d_vec = tl.load(d_base + offs_h).to(tl.float32)
    acc = c_vec[:, None] * d_vec[None, :]

    for k in tl.static_range(BLOCK_HC):
        a_row = tl.load(a_base + k * hc + offs_hco).to(tl.float32)
        b_row = tl.load(b_base + k * hidden + offs_h).to(tl.float32)
        acc += a_row[:, None] * b_row[None, :]

    out_ptrs = x_base + offs_hco[:, None] * hidden + offs_h[None, :]
    tl.store(out_ptrs, acc)


def _select_mhc_post_kernel_meta(
    num_tokens: int,
    hc: int,
    hidden: int,
    n_thr: int,
    h_blk: int,
) -> tuple[int, int, bool, int]:
    block_hc = _next_power_of_two(hc)
    baseline_block_h = min(_next_power_of_two(hidden), _prev_power_of_two(h_blk))
    baseline_programs = num_tokens * triton.cdiv(hidden, baseline_block_h)

    # Policy:
    # - tiny decode: cap hidden tiles at 512 and use full-tile codegen when safe.
    # - normal decode: keep baseline tiles and use full-tile codegen when safe.
    # - large prefill: keep the masked baseline path.
    tiny_decode = baseline_programs <= 16
    decode_grid = baseline_programs <= 256

    if tiny_decode:
        block_h = min(
            _next_power_of_two(hidden),
            _prev_power_of_two(min(h_blk, 512)),
        )
    else:
        block_h = baseline_block_h

    full_tile = hc == block_hc and hidden % block_h == 0
    use_full_tile_kernel = decode_grid and full_tile
    num_warps = max(1, min(8, n_thr // 32))
    return block_hc, block_h, use_full_tile_kernel, num_warps


def mhc_post_triton(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mhc_post_triton")
    if hc <= 0 or hidden <= 0:
        return

    if a.ndim != 3 or b.ndim != 3 or c.ndim != 2 or d.ndim != 2 or x.ndim != 3:
        raise ValueError(
            "mhc_post_triton expects a/b/x rank-3 and c/d rank-2 tensors"
        )
    if a.shape != (a.shape[0], hc, hc):
        raise ValueError(f"Expected a shape (n,{hc},{hc}), got {tuple(a.shape)}")
    if b.shape != x.shape:
        raise ValueError(
            "Expected b and x to have the same shape, "
            f"got {tuple(b.shape)} vs {tuple(x.shape)}"
        )
    if b.shape[1] != hc or b.shape[2] != hidden:
        raise ValueError(
            f"Expected b shape (n,{hc},{hidden}), got {tuple(b.shape)}"
        )
    if c.shape != (a.shape[0], hc):
        raise ValueError(f"Expected c shape (n,{hc}), got {tuple(c.shape)}")
    if d.shape != (a.shape[0], hidden):
        raise ValueError(f"Expected d shape (n,{hidden}), got {tuple(d.shape)}")

    _check_tensor_dtype(
        a, torch.float32, tensor_name="a", fn_name="mhc_post_triton"
    )
    _check_tensor_dtype(
        b, torch.bfloat16, tensor_name="b", fn_name="mhc_post_triton"
    )
    _check_tensor_dtype(
        c, torch.float32, tensor_name="c", fn_name="mhc_post_triton"
    )
    _check_tensor_dtype(
        d, torch.bfloat16, tensor_name="d", fn_name="mhc_post_triton"
    )
    _check_tensor_dtype(
        x, torch.bfloat16, tensor_name="x", fn_name="mhc_post_triton"
    )

    num_tokens = a.shape[0]
    block_hc, block_h, use_full_tile_kernel, num_warps = (
        _select_mhc_post_kernel_meta(num_tokens, hc, hidden, n_thr, h_blk)
    )

    grid = (num_tokens, triton.cdiv(hidden, block_h))
    kernel = (
        _mhc_post_full_tile_triton_kernel
        if use_full_tile_kernel
        else _mhc_post_triton_kernel
    )
    kernel[grid](
        a,
        b,
        c,
        d,
        x,
        hc,
        hidden,
        BLOCK_HC=block_hc,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


# === mhc_post end ===


# === hc_head_fuse begin ===


def _select_hc_head_fuse_kernel_meta(
    num_tokens: int,
    hidden_size: int,
    n_thr: int,
    h_blk: int,
) -> tuple[int, int, int]:
    base_block_h = _prev_power_of_two(h_blk)
    # The kernel masks tails, so this selector only changes tile width and
    # warp count.
    if base_block_h >= 1024:
        # Small batches use 2048 to reduce hidden-loop trips; larger batches use
        # 1024 to limit register pressure. Preserve an explicit larger h_blk.
        small_batch = num_tokens < 48
        target_block_h = 2048 if small_batch else 1024
        block_h_cap = max(base_block_h, target_block_h)
    else:
        block_h_cap = base_block_h
    block_h = min(_next_power_of_two(hidden_size), block_h_cap)

    default_warps = max(1, min(4, n_thr // 128))
    num_warps = 4 if block_h >= 1024 else default_warps
    num_stages = 1
    return block_h, num_warps, num_stages


@triton.jit
def _hc_head_fuse_triton_kernel(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_eps,
    BLOCK_HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)

    offs_hc = tl.arange(0, BLOCK_HC)
    mask_hc = offs_hc < hc_mult

    residual_base = residual_ptr + pid_n * hc_mult * hidden_size
    fn_base = fn_ptr
    out_base = out_ptr + pid_n * hidden_size

    mixes = tl.zeros((BLOCK_HC,), dtype=tl.float32)
    sqrsum = tl.zeros((), dtype=tl.float32)

    for m_c in tl.static_range(BLOCK_HC):
        active_mc = m_c < hc_mult
        for i_h in tl.static_range(NUM_H_BLOCKS):
            offs_h = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < hidden_size
            x_row = tl.load(
                residual_base + m_c * hidden_size + offs_h,
                mask=mask_h & active_mc,
                other=0.0,
            ).to(tl.float32)
            sqrsum += tl.sum(x_row * x_row, axis=0)

            fn_ptrs = (
                fn_base
                + offs_hc[:, None] * hc_mult * hidden_size
                + m_c * hidden_size
                + offs_h[None, :]
            )
            fn_tile = tl.load(
                fn_ptrs,
                mask=mask_h[None, :] & mask_hc[:, None] & active_mc,
                other=0.0,
            ).to(tl.float32)
            mixes += tl.sum(fn_tile * x_row[None, :], axis=1)

    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    hc_base = tl.load(hc_base_ptr + offs_hc, mask=mask_hc, other=0.0).to(
        tl.float32
    )
    rsqrt = tl.rsqrt(sqrsum / tl.cast(hc_mult * hidden_size, tl.float32) + rms_eps)
    pre_mix = tl.sigmoid(mixes * rsqrt * hc_scale + hc_base) + hc_eps

    for i_h in tl.static_range(NUM_H_BLOCKS):
        offs_out = i_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_out = offs_out < hidden_size
        out_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for m_c in tl.static_range(BLOCK_HC):
            active_mc = m_c < hc_mult
            x_row = tl.load(
                residual_base + m_c * hidden_size + offs_out,
                mask=mask_out & active_mc,
                other=0.0,
            ).to(tl.float32)
            pre_scalar = tl.sum(tl.where(offs_hc == m_c, pre_mix, 0.0), axis=0)
            out_acc += pre_scalar * x_row

        tl.store(out_base + offs_out, out_acc, mask=mask_out)


def hc_head_fuse_triton(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hc_head_fuse_triton")
    if hc_mult <= 0 or hidden_size <= 0:
        return

    if (
        residual.ndim != 3
        or fn.ndim != 2
        or hc_scale.ndim != 1
        or hc_base.ndim != 1
        or out.ndim != 2
    ):
        raise ValueError(
            "hc_head_fuse_triton expects residual rank-3, fn rank-2, "
            "hc_scale/hc_base rank-1, out rank-2"
        )

    num_tokens = residual.shape[0]
    residual_shape = residual.shape
    fn_shape = fn.shape
    hc_scale_shape = hc_scale.shape
    hc_base_shape = hc_base.shape
    out_shape = out.shape

    if residual_shape[1] != hc_mult or residual_shape[2] != hidden_size:
        raise ValueError(
            f"Expected residual shape (n,{hc_mult},{hidden_size}), "
            f"got {tuple(residual_shape)}"
        )
    if fn_shape[0] != hc_mult or fn_shape[1] != hc_mult * hidden_size:
        raise ValueError(
            f"Expected fn shape ({hc_mult},{hc_mult * hidden_size}), "
            f"got {tuple(fn_shape)}"
        )
    if hc_scale_shape[0] != 1:
        raise ValueError(
            f"Expected hc_scale shape (1,), got {tuple(hc_scale_shape)}"
        )
    if hc_base_shape[0] != hc_mult:
        raise ValueError(
            f"Expected hc_base shape ({hc_mult},), got {tuple(hc_base_shape)}"
        )
    if out_shape[0] != num_tokens or out_shape[1] != hidden_size:
        raise ValueError(
            f"Expected out shape (n,{hidden_size}), got {tuple(out_shape)}"
        )

    _check_tensor_dtype(
        residual,
        torch.bfloat16,
        tensor_name="residual",
        fn_name="hc_head_fuse_triton",
    )
    _check_tensor_dtype(
        fn, torch.float32, tensor_name="fn", fn_name="hc_head_fuse_triton"
    )
    _check_tensor_dtype(
        hc_scale,
        torch.float32,
        tensor_name="hc_scale",
        fn_name="hc_head_fuse_triton",
    )
    _check_tensor_dtype(
        hc_base,
        torch.float32,
        tensor_name="hc_base",
        fn_name="hc_head_fuse_triton",
    )
    _check_tensor_dtype(
        out,
        torch.bfloat16,
        tensor_name="out",
        fn_name="hc_head_fuse_triton",
    )

    block_hc = _next_power_of_two(hc_mult)
    block_h, num_warps, num_stages = _select_hc_head_fuse_kernel_meta(
        num_tokens, hidden_size, n_thr, h_blk
    )
    num_h_blocks = triton.cdiv(hidden_size, block_h)
    grid = (num_tokens,)
    _hc_head_fuse_triton_kernel[grid](
        residual,
        fn,
        hc_scale,
        hc_base,
        out,
        hc_mult,
        hidden_size,
        rms_eps,
        hc_eps,
        BLOCK_HC=block_hc,
        BLOCK_H=block_h,
        NUM_H_BLOCKS=num_h_blocks,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# === hc_head_fuse end ===


# vLLM custom-op wrappers mirroring
# vllm/model_executor/kernels/mhc/tilelang.py.
# Order intentionally matches the source file:
# op wrapper -> fake helper where the TileLang file defines it.
# `dl_` marks functions registered as torch.ops.vllm.dl_*.


# === vLLM custom op wrappers begin ===


def dl_mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    block_k = 64
    block_m = 64
    n_splits = 1  # DL: force no split-K to match sglang torch path (num_splits=None)
    _ = compute_num_split(
        block_k, hc_hidden_size, math.ceil(num_tokens / block_m)
    )

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    _call_tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    mhc_pre_big_fuse_triton(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _dl_mhc_pre_triton_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def dl_mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)

    out = torch.empty_like(residual)
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return out

    mhc_post_triton(
        comb_res_mix.view(num_tokens, hc_mult, hc_mult),
        residual_flat,
        post_layer_mix.view(num_tokens, hc_mult),
        x.view(num_tokens, hidden_size),
        out.view(num_tokens, hc_mult, hidden_size),
        hc_mult,
        hidden_size,
    )
    return out


# === mhc_fused_post_pre begin ===


def dl_mhc_fused_post_pre_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    fma_token_threshold = 16
    if num_tokens <= fma_token_threshold:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        block_k = 64
        block_m = 64
        n_splits = 1  # DL: force no split-K to match sglang torch path (num_splits=None)
    _ = compute_num_split(
            block_k, hc_hidden_size, math.ceil(num_tokens / block_m)
        )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if num_tokens <= fma_token_threshold:
        mhc_fused_triton(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            split_k=n_splits,
        )
    else:
        mhc_post_triton(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            hc_mult,
            hidden_size,
        )

        _call_tf32_hc_prenorm_gemm(
            residual_cur.view(num_tokens, hc_mult * hidden_size),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )

    mhc_pre_big_fuse_triton(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


# === mhc_fused_post_pre end ===


def _dl_mhc_fused_post_pre_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _dl_mhc_post_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def dl_hc_head_fused_kernel_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    if hs_flat.shape[0] == 0:
        return
    hc_head_fuse_triton(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )


# === vLLM custom op wrappers end ===
