// DL begin: vendored from Denglin vLLM fork csrc/dl/q_gemm_dlblas.cu (port plan 4a GEMM).
// Provides gptq_dlblas_gemmex via dlblas; registered as torch.ops.sgl_kernel.gptq_dlblas_gemmex.
/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "dl/dlblas_helper.cuh"
#include "dlblas_ext.h"

torch::Tensor gptq_dlblas_gemmex(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales, int64_t quant_type, int64_t bit) {
  TORCH_CHECK(
      (a.scalar_type() == c10::ScalarType::Half) ||
          (a.scalar_type() == c10::ScalarType::BFloat16),
      "tensors must be half");
  TORCH_CHECK(
      (b_q_weight.scalar_type() == c10::ScalarType::Float8_e4m3fn) ||
          (b_q_weight.scalar_type() == c10::ScalarType::Byte) ||
          b_q_weight.scalar_type() == c10::ScalarType::Char,
      "qweight must be torch.int8, torch.uint8 or torch.float8_e4m3fn");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());

  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
  dl_cublas_common_args args(a, b_q_weight, c);
  cudaDataType_t kernel_Atype_ =
      dl_scalar_type_to_cuda_data_type(args.mata->scalar_type());
  if (quant_type == 0 && bit == 4) {
    if (b_q_weight.scalar_type() == c10::ScalarType::Char) {
      kernel_Atype_ = static_cast<cudaDataType_t>(CUDA_R_4I);
    } else {
      kernel_Atype_ = static_cast<cudaDataType_t>(CUDA_R_4U);
    }
    args.k = args.k * 2;
  }

  cudaDataType_t kernel_Btype_ =
      dl_scalar_type_to_cuda_data_type(args.matb->scalar_type());
  cudaDataType_t kernel_Ctype_ =
      dl_scalar_type_to_cuda_data_type(c.scalar_type());
  cudaDataType_t computeType_ = static_cast<cudaDataType_t>(CUDA_R_32F);
  cudaDataType_t kernel_Stype_ =
      dl_scalar_type_to_cuda_data_type(b_gptq_scales.scalar_type());
  cudaDataType_t kernel_Qtype_ =
      dl_scalar_type_to_cuda_data_type(b_gptq_qzeros.scalar_type());

  float alpha = 1.0f;
  float beta = 0.0f;
  dlblasExtQuantParametersV2_t extParameters;

  if (quant_type == 0) {
    extParameters.a_group_size_m = args.m / b_gptq_scales.size(1);
    extParameters.a_group_size_k = args.k / b_gptq_scales.size(0);
    extParameters.a_zeropoints_type = kernel_Qtype_;
    extParameters.a_zeropoints = b_gptq_qzeros.data_ptr();
    extParameters.a_scales_type = kernel_Stype_;
    extParameters.a_scales = b_gptq_scales.data_ptr();
  } else if (quant_type == 1) {
    extParameters.a_group_size_m = 1;
    extParameters.a_group_size_k = args.k;
    extParameters.a_zeropoints = nullptr;
    extParameters.a_scales_type = kernel_Stype_;
    extParameters.a_scales = b_gptq_scales.data_ptr();
  } else if (quant_type == 2 || quant_type == 3) {
    int block_shape = 128;
    while ((args.m + block_shape - 1) / block_shape <
           b_gptq_scales.size(0)) {
      block_shape /= 2;
      TORCH_INTERNAL_ASSERT(
          block_shape >= 32,
          "Invalid fp blockwise linear arguments. Weight: [", args.m, ", ",
          args.k, "]. Scales: [", b_gptq_scales.size(0), ", ",
          b_gptq_scales.size(1), "].");
    }
    TORCH_CHECK(
        (args.k + block_shape - 1) / block_shape == b_gptq_scales.size(1));
    extParameters.a_group_size_m = block_shape;
    extParameters.a_group_size_k = block_shape;
    extParameters.a_scales_type = kernel_Stype_;
    extParameters.a_zeropoints = nullptr;
    extParameters.a_scales = b_gptq_scales.data_ptr();
  }

  bool transpose_mat1 = args.transa == 't';
  bool transpose_mat2 = args.transb == 't';
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;

  TORCH_CUDABLAS_CHECK(dlblasGemmExV2(
      at::cuda::getCurrentCUDABlasHandle(), transa, transb, args.m, args.n,
      args.k, &alpha, args.mata->data_ptr(), kernel_Atype_, args.lda,
      args.matb->data_ptr(), kernel_Btype_, args.ldb, &beta,
      args.result->data_ptr(), kernel_Ctype_, args.result_ld, computeType_,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP, &extParameters));
  return c;
}
// DL end
