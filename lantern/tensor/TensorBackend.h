/**
 * @file TensorBackend.h
 * @author Jakob Spahn (jakob@craalse.de)
 * @brief 
 * @date 2022-12-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include "Tensor.h"
#include "Shape.h"

// class Tensor;

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
};

}  // namespace lt
