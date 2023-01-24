#pragma once

#include "lantern/tensor/Tensor.h" 

namespace lt {
namespace cuda {
#define BLOCK_DIM 32
void mm(const Tensor& lhs, const Tensor& rhs, Tensor& out);
void mm_tiled(const Tensor& lhs, const Tensor& rhs, Tensor& out);

void batched_conv2d_hw(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& out);
void batched_conv2d_chw(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& out);
void batched_conv2d_fft(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& out);
}  // namespace cuda
}  // namespace lt