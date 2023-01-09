#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include <cassert>

namespace lt {
CUDABackend& CUDABackend::getInstance() {
    static CUDABackend instance;
    return instance;
}

/******************** ML Operators ********************/
Tensor CUDABackend::reshape(const Tensor& lhs, const Shape& sh) {
    assert(0 && "not implemented");
}

Tensor CUDABackend::transpose(const Tensor& lhs, const Shape& sh) {
    assert(0 && "not implemented");
}

Tensor CUDABackend::matmul(const Tensor& lhs, const Tensor& rhs) {
    assert(0 && "not implemented");
}

Tensor CUDABackend::conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
    assert(0 && "not implemented");
}

Tensor CUDABackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    assert(0 && "not implemented");
}
}