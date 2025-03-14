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


__global__ void add_kernel(data_t const* inp_1, data_t const* inp_2, data_t* outp, dim_t H, dim_t W) {
    dim_t col{blockIdx.x * blockDim.x + threadIdx.x}; 
    dim_t row{blockIdx.y * blockDim.y + threadIdx.y}; 

    if ((row >= H) || (col >= W)) {
        return;
    }

    outp[row*W+col] = inp_1[row*W+col] + inp_2[row*W+col];
}

// Or use fmaxf
__global__ void relu_kernel(data_t const* inp, data_t* outp, dim_t n) {
    dim_t i{blockIdx.x * blockDim.x + threadIdx.x};

    if (i < n) {
        if (inp[i] < 0) {
            outp[i] = 0;
        } else {
            outp[i] = inp[i];
        }
    }
}

__global__ void softmax_kernel(data_t* inp, data_t* outp, dim_t n) {
    dim_t i{blockIdx.x * blockDim.x + threadIdx.x};

    if (i < n) {

        float tmp = inp[0];
        for (int j = 0; j < n; j++) {
            if (inp[j] > tmp) {
                tmp = inp[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += expf(inp[j] - tmp);
        }

        float offset = tmp + logf(sum);
        outp[i] = expf(inp[i] - offset);
    }
}

__global__ void max_pool2d_kernel(data_t* inp, data_t* outp, dim_t input_rows, dim_t input_cols, dim_t kernel_rows, dim_t kernel_cols, dim_t max_val) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int output_row = threadIdx.x;
    int output_col = threadIdx.y;
    int input_row = output_row * kernel_rows;
    int input_col = output_col * kernel_cols;

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols; j++) {
            max_val = fmaxf(
                max_val,
                inp[batch * input_rows * input_cols + channel * input_rows * input_cols + (input_row + i) * input_cols + input_col + j]
            );
        }
    }


    outp[batch * (input_rows / kernel_rows) * (input_cols / kernel_cols) + channel * (input_rows / kernel_rows) * (input_cols / kernel_cols) + output_row * (input_cols / kernel_cols) + output_col] = max_val;
}

/******************** ML Operators ********************/
Tensor CUDABackend::reshape(const Tensor& lhs, const Shape& sh) {
    assert(lhs.shape().elements() == sh.elements());

    return Tensor(std::make_unique<CUDATensor>(
        lhs.getGate<CUDATensor>().data(), 
        sh
    )); // shallow copy with different shape
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
    const dim_t N = lhs.shape()[0], C = lhs.shape()[1], H = lhs.shape()[2], W = lhs.shape()[3];
    const dim_t H_NEW = H / k_sh[0], W_NEW = W / k_sh[0];
    const dim_t stride = k_sh[0];

 
    float max_val = -std::numeric_limits<float>::max();
    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            Shape{N, C, H_NEW, W_NEW})
        );

    dim3 threads_per_block(N, C);
    dim3 blocks_per_grid(H_NEW, W_NEW);
    DBG_PRINT(blocks_per_grid, threads_per_block);

    max_pool2d_kernel<<<threads_per_block, blocks_per_grid>>>(
        lhs.getGate<CUDATensor>().data(),
        ret.getGate<CUDATensor>().data(),
        H,
        W,
        k_sh[0],
        k_sh[1],
        max_val
    );

    cudaDeviceSynchronize();
    
    return ret;
}

Tensor CUDABackend::add(const Tensor& lhs, const Tensor& rhs) {
    checkAddOrThrow(lhs, rhs);
    
    const dim_t H = lhs.shape()[0], W = lhs.shape()[1];

    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            Shape{H, W})
    );

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(W) /
                            static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(H) /
                            static_cast<double>(threads_per_block.y));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    add_kernel<<<blocks_per_grid, threads_per_block>>>(
        lhs.getGate<CUDATensor>().data(),
        rhs.getGate<CUDATensor>().data(),
        ret.getGate<CUDATensor>().data(),
        H,
        W
    );

    cudaDeviceSynchronize();
    return ret;
}

Tensor CUDABackend::relu(const Tensor& lhs) {
    checkReluOrThrow(lhs);

    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            lhs.shape())
    );
    
    dim3 threads_per_block(BLOCK_DIM);
    dim3 blocks_per_grid(1);
    blocks_per_grid.x = std::ceil(static_cast<double>(lhs.elements()) /
                            static_cast<double>(threads_per_block.x));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    relu_kernel<<<blocks_per_grid, threads_per_block>>>(
        lhs.getGate<CUDATensor>().data(),
        ret.getGate<CUDATensor>().data(),
        lhs.elements()
    );

    cudaDeviceSynchronize();

    return ret;
}

Tensor CUDABackend::softmax(const Tensor& lhs) {
    checkSoftmaxOrThrow(lhs);

    auto ptr = lhs.getGate<lt::CUDATensor>().data();
    using PType = std::remove_pointer<decltype(ptr)>::type;
    Tensor ret(
        Tensor::zeros<PType>(
            lhs.shape())
    );

    dim3 threads_per_block(BLOCK_DIM);
    dim3 blocks_per_grid(1);
    blocks_per_grid.x = std::ceil(static_cast<double>(lhs.elements()) /
                            static_cast<double>(threads_per_block.x));
    DBG_PRINT(blocks_per_grid, threads_per_block);
    
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(
        lhs.getGate<CUDATensor>().data(),
        ret.getGate<CUDATensor>().data(),
        lhs.elements()
    );

    cudaDeviceSynchronize();

    return ret;
}

}