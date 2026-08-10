# DL begin
# Denglin (DLIN/DLIN) build for sgl-kernel's common_ops extension.
#
# Mirrors setup_rocm.py (the per-backend setuptools entry), but for DLIN:
# torch's dl-aware torch.utils.cpp_extension detects torch.version.dl and
# drives `dlcc` with --cuda-gpu-arch=dlgput64 automatically (CUDA_HOME must
# point at the DLIN SDK; run via run_sglang.sh build-kernel or with SDK env
# sourced). No CMake / scikit-build — that path can't see dlcc as CUDA.
#
# Source set is a flashinfer-free / CUTLASS-free subset (see common_extension_dl.cc);
# more ops get added in later phases once those headers are available.
#
# Build:  CUDA_HOME=$SDK_DIR python setup_dl.py build_ext --inplace
# (run_sglang.sh build-kernel wraps this.)
import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
    root / "3rdparty" / "shims",  # DL: libcudacxx shim (cuda/functional -> cuda/std/functional)
]

# FlashInfer/CUTLASS-free subset; each entry backs an op registered in
# common_extension_dl.cc. The libcudacxx shim (3rdparty/shims) covers the
# `cuda/functional` include so the moe_topk kernels compile. Grow this set
# further once FlashInfer/CUTLASS headers are vendored (plan §5).
sources = [
    "csrc/common_extension_dl.cc",
    "csrc/elementwise/topk.cu",
    "csrc/elementwise/pos_enc.cu",
    # DL begin: standalone RMSNorm (no FlashInfer dep) — see rmsnorm_dl.cu header.
    "csrc/elementwise/rmsnorm_dl.cu",
    # DL: Gemma RMSNorm (weight+1), replaces vllm._dl_C.gemma_rms_norm (plan 4a).
    "csrc/elementwise/gemma_rmsnorm_dl.cu",
    # DL begin: vendored vllm csrc/dl/ kernels (Phase 4b/4d bulk port).
    # All backed by DL closed-source libs (dlblas/dlblasLt/dldnn).
    "csrc/dl/q_gemm_dlblas.cu",
    "csrc/dl/w8a8_gemm_dlblas.cu",
    "csrc/dl/chunk_gated_delta_rule.cu",
    "csrc/dl/deep_gemm_mqa_logits.cu",
    "csrc/dl/deep_gemm_tf32_hc_prenorm_gemm.cu",
    "csrc/dl/flash_mla_interface.cu",
    "csrc/dl/dl_pos_encoding_kernels.cu",
    "csrc/dl/fused_moe_opt.cu",
    "csrc/dl/dl_invoke_fused_moe_v3.cu",
    "csrc/dl/dl_lora.cu",
    # DL end
    # DL begin: graph-safe paged-decode attention (no packing) — see header.
    "csrc/elementwise/paged_decode_attn_dl.cu",
    # DL end
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    # DL begin: custom allreduce (standard CUDA runtime only — no NVIDIA P2P/IPC;
    # uses fake IPC pointers = pre-allocated SHM). Without this, sglang falls back
    # to slow NCCL (48.5% of decode GPU time per kprof). Ops registered in
    # common_extension_dl.cc.
    "csrc/allreduce/custom_all_reduce.cu",
    # DL end
]

cxx_flags = ["-O3", "-std=c++17"]
# DL begin: torch's CUDAExtension adds -D__CUDA_NO_HALF_CONVERSIONS__ etc by
# default, which breaks static_cast<float>(__half) in vendored vllm DL kernels.
# vllm's CMake build doesn't pass these (it sets only -DENABLE_FP8). Undefine
# them so dlcc allows half/bf16↔float conversions as vllm expects.
nvcc_flags = [
    "-O3", "-std=c++17", "-DENABLE_FP8",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
# DL end
# DLIN CUDA runtime + torch libs (torch's BuildExtension adds c10/torch/...;
# curt is the DLIN libcuda equivalent, resolved via CUDA_HOME=$SDK/lib).
# DL begin: link DL closed-source libs for dlblas-backed kernels (csrc/dl/).
# Headers (dlblas_ext.h / dlblasLt_ext.h / dldnn_ext.h) live in $SDK/include
# (auto-added via CUDA_HOME); libs in $SDK/lib. Mirrors vllm _dl_C's link of
# libdlblasLt.so (CMakeLists:1416).
libraries = ["curt", "dlblas", "dlblasLt", "dldnn"]
# DL end
arch = os.uname().machine
extra_link_args = [f"-L{root}/../.venv/lib/python3.12/site-packages/torch/lib"]

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=[str(p) for p in include_dirs],
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
# DL end
