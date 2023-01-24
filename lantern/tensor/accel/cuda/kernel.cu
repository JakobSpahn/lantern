#include "lantern/tensor/accel/cuda/kernel.cuh"

namespace lt {
namespace cuda {
#define TILE_DIM 32
__global__ void mm_kernel(data_t const* mat_1, data_t const* mat_2, data_t* mat_3, dim_t m,
                          dim_t n, dim_t p)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    dim_t i{blockIdx.x * blockDim.x + threadIdx.x}; // parallel loops through m
    dim_t j{blockIdx.y * blockDim.y + threadIdx.y}; // parallel loops through p

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

__global__ void mm_kernel_tiled(data_t const* mat_1, data_t const* mat_2, data_t* mat_3,
                                dim_t m, dim_t n, dim_t p) {
    __shared__ data_t mat_1_tile[TILE_DIM][TILE_DIM];
    __shared__ data_t mat_2_tile[TILE_DIM][TILE_DIM];

    data_t acc_sum{0};

    for (dim_t tile_idx{0}; tile_idx < ceilf(static_cast<float>(n) / TILE_DIM); ++tile_idx) {
        dim_t i{blockIdx.y * blockDim.y + threadIdx.y}; // loops through m
        dim_t j{tile_idx * blockDim.x + threadIdx.x}; // loops through p
        
        if ((i < m) && (j < n)) {
            mat_1_tile[threadIdx.y][threadIdx.x] = mat_1[i * n + j];
        } else {
            mat_1_tile[threadIdx.y][threadIdx.x] = 0;
        }
        i = tile_idx * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if ((i < n) && (j < p)) {
            mat_2_tile[threadIdx.y][threadIdx.x] = mat_2[i * p + j];
        } else {
            mat_2_tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (dim_t k{0}; k < TILE_DIM; ++k) {
            acc_sum += mat_1_tile[threadIdx.y][k] * mat_2_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    if ((i < m) && (j < p)) {
        mat_3[i * p + j] = acc_sum;
    }
}

// (NCHW)
__global__ void batched_channeled_conv2d_hw_kernel(data_t const* inp, data_t const* kernel, data_t const* bias, data_t* outp,
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
                                    * kernel[(c_out*IN_C*K_HW*K_HW) + (c_in*K_HW*K_HW) + (k_h*K_HW) + (k_w)];  // kernel[c_out, c_in, k_h, k_w]
                    }
                }
            }
            if (bias) {
                acc_sum += bias[c_out];
            }
            outp[(b*OUT_C*H_NEW*W_NEW) + (c_out*H_NEW*W_NEW) + (i*W_NEW) + j] = acc_sum;  // outp[b, c_out, i, j]
        }
    }
}

// (NCHW)
__global__ void batched_channeled_conv2d_chw_kernel(data_t const* inp, data_t const* kernel, data_t const* bias, data_t* outp,
                                     dim_t b_s, dim_t OUT_C, dim_t IN_C, 
                                     dim_t H_OLD, dim_t W_OLD,
                                     dim_t H_NEW, dim_t W_NEW, dim_t K_HW) {
    dim_t c_out{blockIdx.x * blockDim.x + threadIdx.x}; // parallel loops through OUT_C
    dim_t i{blockIdx.y * blockDim.y + threadIdx.y}; // parallel loops through H_NEW
    dim_t j{blockIdx.z * blockDim.z + threadIdx.z}; // parallel loops through W_NEW
    
    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((c_out >= OUT_C) || (i >= H_NEW) || (j >= W_NEW))
    {
        return;
    }

    for (dim_t b{0}; b < b_s; ++b) { // for every batch
        data_t acc_sum{0};
        for (dim_t c_in{0}; c_in < IN_C; ++c_in) {  // for every input channel
            for (dim_t k_h{0}; k_h < K_HW; ++k_h) {
                for (dim_t k_w{0}; k_w < K_HW; ++k_w) {
                    acc_sum += inp[(b*IN_C*H_OLD*W_OLD) + (c_in*H_OLD*W_OLD) + ((i + k_h) * W_OLD) + (j + k_w)] // inp[b, c_in, i+k_h, j+k_w]
                                * kernel[(c_out*IN_C*K_HW*K_HW) + (c_in*K_HW*K_HW) + (k_h*K_HW) + (k_w)];  // kernel[c_out, c_in, k_h, k_w]
                }
            }
        }
        if (bias) {
            acc_sum += bias[c_out];
        }
        outp[(b*OUT_C*H_NEW*W_NEW) + (c_out*H_NEW*W_NEW) + (i*W_NEW) + j] = acc_sum;  // outp[b, c_out, i, j]
    }
}

// Conv2D FFT Kernel
__global__ void conv2dFFTKernel(cufftComplex* mx, cufftComplex* f, cufftComplex* res, dim_t sizeMxX, dim_t sizeF){
    int i = threadIdx.x;
    //printf("Thread%d: Mx-X: %f, Y: %f; F-X: %f, Y: %f\n", i, mx[i].x, mx[i].y, f[i].x, f[i].y);
    res[i].x = mx[i].x * f[i].x - mx[i].y*f[i].y;
    res[i].y = mx[i].x * f[i].y + mx[i].y*f[i].x;
    __syncthreads();
}

}  // namespace cuda
}  // namespace lt