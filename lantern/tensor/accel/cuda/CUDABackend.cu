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
//#define DEBUG

static void DBG_PRINT(const dim3& blocks_per_grid, const dim3& threads_per_block) {
    #ifdef DEBUG
    std::cout << "\nCUDA DEBUG\n\t" 
            << "blocks_per_grid.x=" << blocks_per_grid.x
            << ", blocks_per_grid.y=" << blocks_per_grid.y
            << ", blocks_per_grid.z=" << blocks_per_grid.z
            << ";\n\t"
            << "threads_per_block.x=" << threads_per_block.x
            << ", threads_per_block.y=" << threads_per_block.y
            << ", threads_per_block.z=" << threads_per_block.z
            << std::endl;
    #endif
}

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

// (NCHW)
__global__ void batched_channeled_conv2d_hw_kernel(data_t const* inp, data_t const* kernel, data_t* outp,
                                     dim_t b_s, dim_t OUT_C, dim_t IN_C, 
                                     dim_t H_OLD, dim_t W_OLD,
                                     dim_t H_NEW, dim_t W_NEW, dim_t K_HW) {
    dim_t i{blockIdx.x * blockDim.x + threadIdx.x}; // parallel loops through H_NEW
    dim_t j{blockIdx.y * blockDim.y + threadIdx.y}; // parallel loops through W_NEW
    
    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((i >= H_NEW) || (j >= W_NEW))
    {
        return;
    }

    for (dim_t b{0}; b < b_s; ++b) { // for every batch
        for (dim_t c_out{0}; c_out < OUT_C; ++c_out) {  // for every output channel (this can also be made parallel)
            data_t acc_sum{0};
            for (dim_t c_in{0}; c_in < IN_C; ++c_in) {  // for every input channel
                for (dim_t k_h{0}; k_h < K_HW; ++k_h) {
                    for (dim_t k_w{0}; k_w < K_HW; ++k_w) {
                        acc_sum += inp[(b*IN_C*H_OLD*W_OLD) + (c_in*H_OLD*W_OLD) + ((i + k_h) * W_OLD) + (j + k_w)] // inp[b, c_in, i+k_h, j+k_w]
                                    * kernel[(c_out*IN_C*K_HW*K_HW) + (c_in*K_HW*K_HW) + (k_h*K_HW) + (k_w)];  // kernel[b, c_in, k_h, k_w]
                    }
                }
            }
            outp[(b*OUT_C*H_NEW*W_NEW) + (c_out*H_NEW*W_NEW) + (i*W_NEW) + j] = acc_sum;  // outp[b, c_out, i, j]
        }
    }
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
    DBG_PRINT(blocks_per_grid, threads_per_block);

    mm_kernel<<<blocks_per_grid, threads_per_block>>>(lhs.getGate<lt::CUDATensor>().data(), 
                                                        rhs.getGate<lt::CUDATensor>().data(), 
                                                        ret.getGate<lt::CUDATensor>().data(), m, n, p);
    cudaDeviceSynchronize();

    return ret;
}

Tensor CUDABackend::conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
    checkConv2dOrThrow(lhs, k, b);
    
    const dim_t N = lhs.shape()[0], IN_C = lhs.shape()[1], H_OLD = lhs.shape()[2], W_OLD = lhs.shape()[3],
    OUT_C = k.shape()[0], K_HW = k.shape()[2];
    const dim_t H_NEW = H_OLD - K_HW + 1, W_NEW = W_OLD - K_HW + 1;

    // zero initialize tensor with new shape (N,OUT_C,H_NEW,W_NEW)
    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            Shape{N, OUT_C, H_NEW, W_NEW})
        );

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(H_NEW) /
                            static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(W_NEW) /
                            static_cast<double>(threads_per_block.y));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    batched_channeled_conv2d_hw_kernel<<<blocks_per_grid, threads_per_block>>>(
                            lhs.getGate<CUDATensor>().data(), 
                            k.getGate<CUDATensor>().data(), 
                            ret.getGate<CUDATensor>().data(),
                            N, OUT_C, IN_C, 
                            H_OLD, W_OLD,
                            H_NEW, W_NEW, K_HW);
    cudaDeviceSynchronize();

    return ret;
}

Tensor CUDABackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    checkMaxPoolOrThrow(lhs, k_sh);
    assert(0 && "not implemented");
    return Tensor();
}

// TODO: Not yet implemented
Tensor CUDABackend::add(const Tensor& lhs, const Tensor& rhs) {
    checkAddOrThrow(lhs, rhs);
    assert(0 && "not implemented");
    return Tensor();
}

}