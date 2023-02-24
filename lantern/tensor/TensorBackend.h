#pragma once

#include "lantern/tensor/Tensor.h"
#include "lantern/tensor/Shape.h"

namespace lt {

class TensorBackend {
 public:
    TensorBackend() = default;
    virtual ~TensorBackend() = default;

    /******************** ML Operators ********************/
    virtual Tensor reshape(const Tensor& lhs, const Shape& sh) = 0;
    virtual Tensor transpose(const Tensor& lhs, const Shape& sh) = 0;
    virtual Tensor matmul(const Tensor& lhs, const Tensor& rhs) = 0;
    virtual Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) = 0;
    virtual Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh) = 0;
    virtual Tensor add(const Tensor& lhs, const Tensor& rhs) = 0;
    virtual Tensor relu(const Tensor& lhs) = 0;
    virtual Tensor softmax(const Tensor& lhs) = 0;
};

}  // namespace lt
