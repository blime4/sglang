#pragma once

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>
#include <torch/all.h>

#define TORCH_CUDABLAS_CHECK(EXPR)                              \
  do {                                                          \
    cublasStatus_t __err = EXPR;                                \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CUDA error: ",                                 \
                at::cuda::blas::_cublasGetErrorEnum(__err),     \
                " when calling `" #EXPR "`");                   \
  } while (0)

inline cudaDataType_t dl_scalar_type_to_cuda_data_type(
    const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return CUDA_R_8U;
    case c10::ScalarType::Char:
      return CUDA_R_8I;
    case c10::ScalarType::Int:
      return CUDA_R_32I;
    case c10::ScalarType::Half:
      return CUDA_R_16F;
    case c10::ScalarType::Float:
      return CUDA_R_32F;
    case c10::ScalarType::Double:
      return CUDA_R_64F;
    case c10::ScalarType::ComplexHalf:
      return CUDA_C_16F;
    case c10::ScalarType::ComplexFloat:
      return CUDA_C_32F;
    case c10::ScalarType::ComplexDouble:
      return CUDA_C_64F;
    case c10::ScalarType::Short:
      return CUDA_R_16I;
    case c10::ScalarType::Long:
      return CUDA_R_64I;
    case c10::ScalarType::BFloat16:
      return CUDA_R_16BF;
    case c10::ScalarType::Float8_e4m3fn:
      return static_cast<cudaDataType_t>(CUDA_R_8F_E4M3);
    case c10::ScalarType::Float8_e5m2:
      return static_cast<cudaDataType_t>(CUDA_R_8F_E5M2);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Cannot convert ScalarType ", scalar_type,
          " to cudaDataType.");
  }
}

inline c10::MaybeOwned<torch::Tensor> dl_resolve_conj_if_indicated(
    const torch::Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<torch::Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<torch::Tensor>::borrowed(tensor);
  }
}

inline c10::MaybeOwned<torch::Tensor> dl_prepare_matrix_for_cublas(
    const torch::Tensor& tensor, bool& transpose_tensor,
    bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) {
    transpose_tensor = tensor.is_contiguous();
    return dl_resolve_conj_if_indicated(
        tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  torch::IntArrayRef tensor_strides = tensor.strides();
  torch::IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return dl_resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) &&
             (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return dl_resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<torch::Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

inline c10::MaybeOwned<torch::Tensor> dl_prepare_matrix_for_cublas(
    const torch::Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    transpose_tensor = tensor.is_contiguous();
    return dl_resolve_conj_if_indicated(tensor, true);
  }

  torch::IntArrayRef tensor_strides = tensor.strides();
  torch::IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return dl_resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) &&
             (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return dl_resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<torch::Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

struct dl_cublas_common_args {
  dl_cublas_common_args(
      const torch::Tensor& mat1, const torch::Tensor& mat2, torch::Tensor& c) {
    bool transpose_result = false;
    bool transpose_mat1 = false;
    bool transpose_mat2 = false;
    result = dl_prepare_matrix_for_cublas(c, transpose_result);
    mata = dl_prepare_matrix_for_cublas(
        transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = dl_prepare_matrix_for_cublas(
        transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_mat1 ? mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_mat2 ? matb->is_conj() ? 'c' : 't' : 'n';
  }

  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<torch::Tensor> mata, matb, result;
};
