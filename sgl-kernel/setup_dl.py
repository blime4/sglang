# DL begin
# Denglin (登临/DLIN) build for sgl-kernel's common_ops extension.
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
include_dirs = [root / "include", root / "include" / "impl", root / "csrc"]

# FlashInfer/CUTLASS/libcudacxx-free subset; each entry backs an op registered
# in common_extension_dl.cc. Grow this set in later phases (needs FetchContent).
sources = [
    "csrc/common_extension_dl.cc",
    "csrc/elementwise/topk.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/memory/weak_ref_tensor.cpp",
]

cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17"]
# DLIN CUDA runtime + torch libs (torch's BuildExtension adds c10/torch/...;
# curt is the DLIN libcuda equivalent, resolved via CUDA_HOME=$SDK/lib).
libraries = ["curt"]
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
