#pragma once

#include "lantern/tensor/TensorBackend.h"

namespace lt {

class CUDABackend : public lt::TensorBackend {
 public:
    CUDABackend() = default;
    ~CUDABackend() = default;

    static CUDABackend& getInstance(); 

    // this needs to be done better
    bool tile = false;
    bool conv_use_chw = false;
	bool conv_fft = false;

    /******************** ML Operators ********************/
    Tensor reshape(const Tensor& lhs, const Shape& sh);
    Tensor transpose(const Tensor& lhs, const Shape& sh);
    Tensor matmul(const Tensor& lhs, const Tensor& rhs);
    Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b);
    Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh);
};

}  // namespace lt