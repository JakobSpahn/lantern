#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/RuntimeCheck.h"
#include "lantern/tensor/accel/cuda/blas.h"

#include <cassert>
#include <cmath>
#include <type_traits>
#include <iostream>

namespace lt {
CUDABackend& CUDABackend::getInstance() {
    static CUDABackend instance;
    return instance;
}

/******************** ML Operators ********************/
Tensor CUDABackend::reshape(const Tensor& lhs, const Shape& sh) {
    assert(0 && "not implemented");
    return Tensor();
}

Tensor CUDABackend::transpose(const Tensor& lhs, const Shape& sh) {
    assert(0 && "not implemented");
    return Tensor();
}

Tensor CUDABackend::matmul(const Tensor& lhs, const Tensor& rhs) {
    checkMatmulOrThrow(lhs, rhs);

    // get zero initialized result tensor
    Tensor ret(Tensor::zeros<data_t>({lhs.shape()[0], rhs.shape()[1]}));

    if (!tile) {
        cuda::mm(lhs, rhs, ret);
    } else {
        cuda::mm_tiled(lhs, rhs, ret);
    }
     
    return ret;
}

Tensor CUDABackend::conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
    checkConv2dOrThrow(lhs, k, b);
    
    const dim_t N = lhs.shape()[0], IN_C = lhs.shape()[1], H_OLD = lhs.shape()[2], W_OLD = lhs.shape()[3],
    OUT_C = k.shape()[0], K_HW = k.shape()[2];
    const dim_t H_NEW = H_OLD - K_HW + 1, W_NEW = W_OLD - K_HW + 1;

    // zero initialize tensor with new shape (N,OUT_C,H_NEW,W_NEW)
    Tensor ret(Tensor::zeros<data_t>({N, OUT_C, H_NEW, W_NEW}));

    if (!conv_use_chw && !conv_fft) {
        cuda::batched_conv2d_hw(lhs, k, b, ret);
    } else if (conv_use_chw){
        cuda::batched_conv2d_chw(lhs, k, b, ret);
    } else if (!conv_use_chw && conv_fft) {
		std::cout << "fft" << std::endl;
		cuda::batched_conv2d_fft(lhs, k, b, ret);
	}

    return ret;
}

Tensor CUDABackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    checkMaxPoolOrThrow(lhs, k_sh);
    assert(0 && "not implemented");
    return Tensor();
}
}