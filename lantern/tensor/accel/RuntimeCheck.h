#pragma once

#include "lantern/tensor/Tensor.h"

namespace lt {
void checkAddOrThrow(const Tensor& a, const Tensor& b);
void checkMatmulOrThrow(const Tensor& a, const Tensor& b);
void checkConv2dOrThrow(const Tensor& a, const Tensor& f, const Tensor& b);
void checkMaxPoolOrThrow(const Tensor& a, const Shape& k_sh);
void checkReluOrThrow(const Tensor& a);
void checkSoftmaxOrThrow(const Tensor& b);
}  // namespace lt