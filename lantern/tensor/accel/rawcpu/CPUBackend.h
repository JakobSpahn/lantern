#pragma once

#include "lantern/tensor/TensorBackend.h"

namespace lt {

class CPUBackend : public lt::TensorBackend {
 public:
    CPUBackend() = default;
    ~CPUBackend() = default;

    static CPUBackend& getInstance(); 

    /******************** ML Operators ********************/
    Tensor reshape(const Tensor& lhs, const Shape& sh);
    Tensor transpose(const Tensor& lhs, const Shape& sh);
    Tensor matmul(const Tensor& lhs, const Tensor& rhs);
    Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b);
    Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh);
};

}  // namespace lt