#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/RuntimeCheck.h"

#include <cassert>
#include <cmath>
#include <type_traits>
#include <iostream>

namespace lt {
CUDABackend& CUDABackend::getInstance() {
    static CUDABackend instance;
    return instance;
}

#define BLOCK_DIM 32
#define dbg "\nCUDA DEBUG: "

__global__ void mm_kernel(data_t const* mat_1, data_t const* mat_2, data_t* mat_3, size_t m,
                          size_t n, size_t p)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y}; // loops through m
    size_t j{blockIdx.x * blockDim.x + threadIdx.x}; // loops through p

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((i >= m) || (j >= p))
    {
        return;
    }

    data_t acc_sum{0};
    for (size_t k{0}; k < n; ++k)
    {
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
    }
    mat_3[i * p + j] = acc_sum;
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
    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            Shape{lhs.shape()[0], rhs.shape()[1]})
        );
    size_t m(lhs.shape()[0]), n(lhs.shape()[1]), p(rhs.shape()[1]);

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));

    std::cout << dbg 
            << "matmul with blocks_per_grid.x=" << blocks_per_grid.x
            << ", blocks_per_grid.y=" << blocks_per_grid.y
            << "; threads_per_block.x=" << threads_per_block.x
            << ", threads_per_block.y=" << threads_per_block.y
            << std::endl;

    mm_kernel<<<blocks_per_grid, threads_per_block>>>(lhs.getGate<lt::CUDATensor>().data(), 
                                                        rhs.getGate<lt::CUDATensor>().data(), 
                                                        ret.getGate<lt::CUDATensor>().data(), m, n, p);
    cudaDeviceSynchronize();

    return ret;
}

Tensor CUDABackend::conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
    checkConv2dOrThrow(lhs, k, b);
    assert(0 && "not implemented");
    return Tensor();
}

Tensor CUDABackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    checkMaxPoolOrThrow(lhs, k_sh);
    assert(0 && "not implemented");
    return Tensor();
}
}