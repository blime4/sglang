#!/usr/bin/env python3
# Unit test for the DLIN sgl_kernel rmsnorm / fused_add_rmsnorm (rmsnorm_dl.cu).
# Compares the dlcc kernel against a pure-Python reference (the same math as
# RMSNorm.forward_native) on random bf16/fp16 inputs. Needs no model / no decode.
#
# Run:  source .venv/bin/activate  (with SDK_DIR=.../sdk env active)
#       python scripts/dl/test_rmsnorm_dl.py
import torch
import sgl_kernel  # noqa: F401  loads common_ops.so -> registers torch.ops.sgl_kernel.*

torch.manual_seed(0)
HIDDEN = 2048  # Qwen3-1.7B hidden_size
CASES = [(1, HIDDEN), (4, HIDDEN), (32, HIDDEN), (7, 4096)]


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    var = (xf * xf).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * w.float()
    return out.to(x.dtype)


def ref_fused_add_rmsnorm(x, residual, w, eps):
    s = (x.float() + residual.float())
    residual_out = s.to(x.dtype)
    var = (s * s).mean(-1, keepdim=True)
    out = s * torch.rsqrt(var + eps) * w.float()
    return out.to(x.dtype), residual_out


def main():
    dev = "cuda"
    for dtype in (torch.bfloat16, torch.float16):
        for shape in CASES:
            eps = 1e-6
            x = torch.randn(shape, dtype=dtype, device=dev) * 0.5
            w = torch.randn(shape[-1], dtype=dtype, device=dev) * 0.1

            # --- rmsnorm ---
            out = torch.empty_like(x)
            torch.ops.sgl_kernel.rmsnorm.default(out, x, w, eps, False)
            ref = ref_rmsnorm(x, w, eps)
            max_err = (out.float() - ref.float()).abs().max().item()
            ok = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
            print(f"rmsnorm     {str(dtype):16s} {str(shape):14s} max_err={max_err:.4e}  {'OK' if ok else 'FAIL'}")

            # --- fused_add_rmsnorm ---
            x2 = torch.randn(shape, dtype=dtype, device=dev) * 0.5
            res = torch.randn(shape, dtype=dtype, device=dev) * 0.3
            x2_k = x2.clone()
            res_k = res.clone()
            torch.ops.sgl_kernel.fused_add_rmsnorm.default(x2_k, res_k, w, eps, False)
            ref_out, ref_res = ref_fused_add_rmsnorm(x2, res, w, eps)
            err_out = (x2_k.float() - ref_out.float()).abs().max().item()
            err_res = (res_k.float() - ref_res.float()).abs().max().item()
            ok_out = torch.allclose(x2_k.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
            ok_res = torch.allclose(res_k.float(), ref_res.float(), atol=2e-2, rtol=2e-2)
            print(
                f"fused_add   {str(dtype):16s} {str(shape):14s} "
                f"err_out={err_out:.4e} err_res={err_res:.4e}  {'OK' if ok_out and ok_res else 'FAIL'}"
            )

    # error-path: fp32 should be rejected (kernel supports fp16/bf16 only).
    try:
        x = torch.randn(2, HIDDEN, dtype=torch.float32, device=dev)
        w = torch.randn(HIDDEN, dtype=torch.float32, device=dev)
        out = torch.empty_like(x)
        torch.ops.sgl_kernel.rmsnorm.default(out, x, w, 1e-6, False)
        print("fp32 reject: FAIL (expected TORCH_CHECK)")
    except RuntimeError as e:
        print(f"fp32 reject: OK (raised: {str(e)[:60]})")


if __name__ == "__main__":
    main()
