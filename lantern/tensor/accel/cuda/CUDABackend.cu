#include "lantern/tensor/accel/cuda/CUDABackend.cuh"

#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/RuntimeCheck.h"

#include <cassert>
#include <cmath>
#include <type_traits>
#include <iostream>

namespace lt {
//#define DEBUG
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



}