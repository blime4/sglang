// DL begin: vendored from Denglin vLLM fork csrc/dl/w8a8_gemm_dlblas.cu (port plan 4b/4d).
/*
 * W8A8 / W8AFloat Quantized GEMM using dlblasLtMatmul.
 */

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "dl/dlblas_helper.cuh"
#include "dlblasLt_ext.h"

namespace {

inline bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  for (char& ch : normalized) {
    ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
  }
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

inline void ValidateInputs(torch::Tensor a, torch::Tensor b_q_weight,
                           std::optional<torch::Tensor> const& a_scales,
                           torch::Tensor b_scales, bool a_is_quantized,
                           bool require_a_scales) {
  TORCH_CHECK((a.scalar_type() == c10::ScalarType::Half) ||
                  (a.scalar_type() == c10::ScalarType::BFloat16) ||
                  (a_is_quantized &&
                   a.scalar_type() == c10::ScalarType::Char),
              "Activation tensor must be half, bfloat16, or int8 "
              "(if a_is_quantized=true)");
  TORCH_CHECK(b_q_weight.scalar_type() == c10::ScalarType::Char,
              "Weight tensor must be int8");
  TORCH_CHECK(a.dim() == 2 && b_q_weight.dim() == 2,
              "Input tensors must be 2-dimensional");
  TORCH_CHECK(a.size(1) == b_q_weight.size(1),
              "K dimension must match: A.size(1) must equal B.size(1)");
  TORCH_CHECK(b_q_weight.size(0) == b_scales.size(0),
              "N dimension must match: B.size(0) must equal b_scales.size(0)");
  TORCH_CHECK(b_scales.scalar_type() == c10::ScalarType::Float,
              "b_scales must be float32");
  if (require_a_scales && a_is_quantized) {
    TORCH_CHECK(a_scales.has_value(),
                "a_scales must be provided when a_is_quantized=true");
    TORCH_CHECK(a_scales->scalar_type() == c10::ScalarType::Float,
                "a_scales must be float32");
    TORCH_CHECK(a_scales->dim() == 1 || a_scales->dim() == 2,
                "a_scales must be a 1D or 2D tensor");
    TORCH_CHECK(a_scales->size(0) == a.size(0),
                "a_scales.size(0) must equal A.size(0)");
  }
}

inline cublasLtMatrixLayout_t CreateScaleLayout(int64_t rows) {
  cublasLtMatrixLayout_t scale_layout;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&scale_layout, CUDA_R_32F, rows, 1, 1));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      scale_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  return scale_layout;
}

}  // namespace

torch::Tensor w8a8_matmul_advance(torch::Tensor a, torch::Tensor b_q_weight,
                                  std::optional<torch::Tensor> const& a_scales,
                                  torch::Tensor b_scales,
                                  bool a_is_quantized) {
  ValidateInputs(a, b_q_weight, a_scales, b_scales, a_is_quantized, true);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  auto output_dtype = a_is_quantized ? c10::ScalarType::Half : a.scalar_type();
  auto options = torch::TensorOptions().dtype(output_dtype).device(a.device());
  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(0)}, options);

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b_q_weight.size(0);

  cudaDataType_t a_type =
      a_is_quantized ? CUDA_R_8I
                     : dl_scalar_type_to_cuda_data_type(a.scalar_type());
  cudaDataType_t c_type = dl_scalar_type_to_cuda_data_type(c.scalar_type());

  cublasLtMatrixLayout_t a_desc, b_desc, c_desc, d_desc;
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, a_type, m, k, k));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, n, k, k));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, n));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, c_type, m, n, n));

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  cublasLtMatrixLayout_t scale_a_layout = CreateScaleLayout(m);
  dlblasLtQuantParamsConfig_t aq_config;
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigCreate(&aq_config));
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigSet(
      aq_config, scale_a_layout,
      a_is_quantized && a_scales.has_value() ? a_scales->data_ptr() : nullptr,
      nullptr, nullptr,
      a_is_quantized ? DLBLASLT_QUANT_OP_DEQUANTIZE
                     : DLBLASLT_QUANT_OP_QUANTIZE));

  cublasLtMatrixLayout_t scale_b_layout = CreateScaleLayout(n);
  dlblasLtQuantParamsConfig_t bq_config;
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigCreate(&bq_config));
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigSet(
      bq_config, scale_b_layout, b_scales.data_ptr(), nullptr, nullptr,
      DLBLASLT_QUANT_OP_DEQUANTIZE));

  cublasLtMatmulDesc_t matmul_desc;
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32F));

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  size_t workspace_size_in_bytes = 0;
  TORCH_CUDABLAS_CHECK(dlblasLtMatmulGetWorkspace(
      matmul_desc, a_desc, aq_config, b_desc, bq_config, c_desc, nullptr,
      d_desc, nullptr, &workspace_size_in_bytes));

  at::Tensor workspace_tensor;
  void* workspace_ptr = nullptr;
  if (workspace_size_in_bytes > 0) {
    workspace_tensor = at::empty(
        {static_cast<int64_t>(workspace_size_in_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    workspace_ptr = workspace_tensor.data_ptr();
  }

  int32_t alpha = 1;
  int32_t beta = 0;

  cublasStatus_t result = dlblasLtMatmul(
      at::cuda::getCurrentCUDABlasLtHandle(), matmul_desc, &alpha, a.data_ptr(),
      a_desc, aq_config, b_q_weight.data_ptr(), b_desc, bq_config, &beta,
      nullptr, c_desc, nullptr, c.data_ptr(), d_desc, nullptr, nullptr,
      workspace_ptr, workspace_size_in_bytes,
      at::cuda::getCurrentCUDAStream());

  TORCH_CUDABLAS_CHECK(result);

  dlblasLtQuantParamsConfigDestroy(aq_config);
  dlblasLtQuantParamsConfigDestroy(bq_config);
  cublasLtMatrixLayoutDestroy(scale_a_layout);
  cublasLtMatrixLayoutDestroy(scale_b_layout);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(d_desc);
  cublasLtMatmulDescDestroy(matmul_desc);
  return c;
}

namespace {

torch::Tensor w8a8_matmul_float_a(torch::Tensor a, torch::Tensor b_q_weight,
                                  std::optional<torch::Tensor> const& a_scales,
                                  torch::Tensor b_scales,
                                  bool a_is_quantized) {
  ValidateInputs(a, b_q_weight, a_scales, b_scales, a_is_quantized, false);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  auto output_dtype = a_is_quantized ? c10::ScalarType::Half : a.scalar_type();
  auto options = torch::TensorOptions().dtype(output_dtype).device(a.device());
  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(0)}, options);

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b_q_weight.size(0);

  cudaDataType_t a_type =
      a_is_quantized ? CUDA_R_8I
                     : dl_scalar_type_to_cuda_data_type(a.scalar_type());
  cudaDataType_t c_type = dl_scalar_type_to_cuda_data_type(c.scalar_type());

  cublasLtMatrixLayout_t a_desc, b_desc, c_desc, d_desc;
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, a_type, m, k, k));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, n, k, k));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, n));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, c_type, m, n, n));

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  cublasLtMatrixLayout_t scale_b_layout = CreateScaleLayout(n);
  dlblasLtQuantParamsConfig_t bq_config;
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigCreate(&bq_config));
  TORCH_CUDABLAS_CHECK(dlblasLtQuantParamsConfigSet(
      bq_config, scale_b_layout, b_scales.data_ptr(), nullptr, nullptr,
      DLBLASLT_QUANT_OP_DEQUANTIZE));

  cublasLtMatmulDesc_t matmul_desc;
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t result = dlblasLtMatmul(
      at::cuda::getCurrentCUDABlasLtHandle(), matmul_desc, &alpha, a.data_ptr(),
      a_desc, nullptr, b_q_weight.data_ptr(), b_desc, bq_config, &beta,
      nullptr, c_desc, nullptr, c.data_ptr(), d_desc, nullptr, nullptr,
      nullptr, 0, at::cuda::getCurrentCUDAStream());

  TORCH_CUDABLAS_CHECK(result);

  dlblasLtQuantParamsConfigDestroy(bq_config);
  cublasLtMatrixLayoutDestroy(scale_b_layout);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(d_desc);
  cublasLtMatmulDescDestroy(matmul_desc);
  return c;
}

}  // namespace

torch::Tensor w8a8_matmul(torch::Tensor a, torch::Tensor b_q_weight,
                          std::optional<torch::Tensor> const& a_scales,
                          torch::Tensor b_scales,
                          bool a_is_quantized) {
  if (EnvEnabled("VLLM_W8A8_MATMUL_ADVANCE")) {
      return w8a8_matmul_float_a(a, b_q_weight, 
                                 a_scales, 
                                 b_scales,
                                 a_is_quantized);
  }

  return w8a8_matmul_advance(a, b_q_weight, 
                  a_scales, 
                  b_scales,
                  a_is_quantized);
}
// DL end
