#pragma once

#include "lantern/tensor/accel/cuda/CUDATensor.h" 
#include "lantern/tensor/accel/cuda/kernel.cuh"

#include <cufft.h>

namespace lt {
namespace cuda {
__global__ void mm_kernel(data_t const* mat_1, data_t const* mat_2, data_t* mat_3, dim_t m,
                            dim_t n, dim_t p);
__global__ void mm_kernel_tiled(data_t const* mat_1, data_t const* mat_2, data_t* mat_3,
                                dim_t m, dim_t n, dim_t p);

__global__ void batched_channeled_conv2d_hw_kernel(data_t const* inp, data_t const* kernel, data_t const* bias, data_t* outp,
                                     dim_t b_s, dim_t OUT_C, dim_t IN_C, 
                                     dim_t H_OLD, dim_t W_OLD,
                                     dim_t H_NEW, dim_t W_NEW, dim_t K_HW);
__global__ void batched_channeled_conv2d_chw_kernel(data_t const* inp, data_t const* kernel, data_t const* bias, data_t* outp,
                                        dim_t b_s, dim_t OUT_C, dim_t IN_C, 
                                        dim_t H_OLD, dim_t W_OLD,
                                        dim_t H_NEW, dim_t W_NEW, dim_t K_HW);

__global__ void conv2dFFTKernel(cufftComplex* mx, cufftComplex* f, cufftComplex* res, dim_t sizeMxX, dim_t sizeF);
}  // namespace cuda
}  // namespace lt