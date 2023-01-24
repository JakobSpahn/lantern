#include "lantern/tensor/accel/cuda/blas.h"

#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/cuda/kernel.cuh"

#include <cassert>
#include <cufft.h>

namespace lt {
namespace cuda {
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

void mm(const Tensor& lhs, const Tensor& rhs, Tensor& ret) {
    size_t m(lhs.shape()[0]), n(lhs.shape()[1]), p(rhs.shape()[1]);

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.y));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    mm_kernel<<<blocks_per_grid, threads_per_block>>>(lhs.getGate<lt::CUDATensor>().data(), 
                                                        rhs.getGate<lt::CUDATensor>().data(), 
                                                        ret.getGate<lt::CUDATensor>().data(), m, n, p);
    cudaDeviceSynchronize();
}

void mm_tiled(const Tensor& lhs, const Tensor& rhs, Tensor& ret) {
    size_t m(lhs.shape()[0]), n(lhs.shape()[1]), p(rhs.shape()[1]);

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    mm_kernel_tiled<<<blocks_per_grid, threads_per_block>>>(lhs.getGate<lt::CUDATensor>().data(), 
                                                        rhs.getGate<lt::CUDATensor>().data(), 
                                                        ret.getGate<lt::CUDATensor>().data(), m, n, p);
    cudaDeviceSynchronize();
}
    
void batched_conv2d_hw(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& ret) {
    const dim_t N = lhs.shape()[0], IN_C = lhs.shape()[1], H_OLD = lhs.shape()[2], W_OLD = lhs.shape()[3],
    OUT_C = k.shape()[0], K_HW = k.shape()[2];
    const dim_t H_NEW = H_OLD - K_HW + 1, W_NEW = W_OLD - K_HW + 1;

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
                            (b.isEmpty() ? nullptr : b.getGate<CUDATensor>().data()),
                            ret.getGate<CUDATensor>().data(),
                            N, OUT_C, IN_C, 
                            H_OLD, W_OLD,
                            H_NEW, W_NEW, K_HW);
    cudaDeviceSynchronize();
}

void batched_conv2d_chw(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& ret) {
    const dim_t N = lhs.shape()[0], IN_C = lhs.shape()[1], H_OLD = lhs.shape()[2], W_OLD = lhs.shape()[3],
    OUT_C = k.shape()[0], K_HW = k.shape()[2];
    const dim_t H_NEW = H_OLD - K_HW + 1, W_NEW = W_OLD - K_HW + 1;

    dim3 threads_per_block(10, 10, 10);
    dim3 blocks_per_grid(1, 1);

    blocks_per_grid.x = std::ceil(static_cast<double>(OUT_C) /
                            static_cast<double>(threads_per_block.x)); 
    blocks_per_grid.y = std::ceil(static_cast<double>(H_NEW) /
                            static_cast<double>(threads_per_block.y));
    blocks_per_grid.z = std::ceil(static_cast<double>(W_NEW) /
                            static_cast<double>(threads_per_block.z));
    DBG_PRINT(blocks_per_grid, threads_per_block);

    batched_channeled_conv2d_chw_kernel<<<blocks_per_grid, threads_per_block>>>(
                            lhs.getGate<CUDATensor>().data(), 
                            k.getGate<CUDATensor>().data(), 
                            (b.isEmpty() ? nullptr : b.getGate<CUDATensor>().data()),
                            ret.getGate<CUDATensor>().data(),
                            N, OUT_C, IN_C, 
                            H_OLD, W_OLD,
                            H_NEW, W_NEW, K_HW);
    cudaDeviceSynchronize();
}

void batched_conv2d_fft(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& out) {
    /*
    //Assertion: Filter is padded with zeroes to size of MX
    //Assert Filter has unpadded size 5
    assert(sizeF == 5);

    int resSizeX = sizeMxX - sizeF + 1;
    int resSizeY = sizeMxY - sizeF + 1;

    cufftReal* dev_mxR = 0;
    cufftReal* dev_fR = 0;
    cufftReal* dev_resR = 0;
    cufftComplex* dev_mxC = 0;
    cufftComplex* dev_fC = 0;
    cufftComplex* dev_resC = 0;


    // CUFFT plan  
    cufftHandle planMx;
    cufftHandle planRes;
    cufftPlan2d(&planMx, sizeMxX, sizeMxY, CUFFT_R2C);
    cufftPlan2d(&planRes, sizeMxX, sizeMxY, CUFFT_C2R);
     
    // Transform signal and filter
    cufftExecR2C(planMx, (cufftReal *)dev_mxR, (cufftComplex *)dev_mxC);
    cufftExecR2C(planMx, (cufftReal *)dev_fR, (cufftComplex *)dev_fC);

    conv2dFFTKernel<<<1, sizeMxX*sizeMxY>>>(dev_mxC, dev_fC, dev_resC, sizeMxX, 5);

    // Transform signal back
    cufftExecC2R(planRes, (cufftComplex *)dev_resC, (cufftReal *)dev_resR);
    */
}
}  // namespace cuda
}  // namespace lt