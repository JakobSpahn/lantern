#include "lantern/tensor/accel/cuda/blas.h"

#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/cuda/kernel.cuh"

#include <cassert>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <iostream>

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

static data_t& get(const Tensor& t, const Shape& idx) {
	assert(t.ndim() == idx.ndim() && "Number of indices doesn't match shape of tensor");

	size_t max_dims = idx.ndim();
	dim_t n = 0;

#pragma unroll
	for (dim_t i = 0; i < idx.ndim(); ++i) {
		assert(idx[i] < t.shape()[i] && "Access index out of bounds");

		Shape shp(t.shape());
		if (i < (max_dims - 1)) [[likely]] {
			n += idx[i] * std::accumulate(shp.get().cbegin() + i + 1, shp.get().cend(), 1, std::multiplies<dim_t>()) ;
		} else {
			n += idx[i];
		}
	}

	return t.getGate<CUDATensor>().data()[n];
}

void batched_conv2d_fft(const Tensor& lhs, const Tensor& k, const Tensor& b, Tensor& out) {
	assert(lhs.shape()[0] == lhs.shape()[1] == 1 && "only works for 2d tensors"); // for now
	const long long H{lhs.shape()[2]},
					W{lhs.shape()[3]},
					K_HW{k.shape()[2]};
    //Assertion: Filter is padded with zeroes to size of MX
	// pad the kernel
	Tensor k_padded(Tensor::zeros<float>(Shape{k.shape()[0], k.shape()[1], H, W}));
	for (long long i{0}; i < k.shape()[0]; ++i) {
		for (long long j{0}; j < k.shape()[1]; ++j) {
			for (long long l{0}; l < k.shape()[2]; ++l) {
				for (long long m{0}; m < k.shape()[3]; ++m) {
					const Shape idx{i,j,l,m};
					get(k_padded, idx) = get(k, idx);
				}
			}
		}
	}

    //Assert Filter has unpadded size 5
    assert(K_HW == 5);

    float* inpReal = nullptr;
    cufftReal* kernReal = nullptr;
    cufftReal* outReal = out.getGate<CUDATensor>().data();
    cufftComplex* inpCplx = nullptr;
    cufftComplex* kernCplx = nullptr;
    cufftComplex* outCplx = nullptr;

	// copy tensor to inpReal
	cudaMallocManaged(reinterpret_cast<void **>(&inpReal), lhs.elements() * sizeof(cufftReal));
	cudaMemcpy(inpReal, lhs.getGate<CUDATensor>().data(), lhs.elements() * sizeof(cufftReal), cudaMemcpyDefault);

	// copy kernel to kernelReal
	cudaMallocManaged(reinterpret_cast<void **>(&kernReal), k_padded.elements() * sizeof(cufftReal));
	cudaMemcpy(kernReal, k_padded.getGate<CUDATensor>().data(), k_padded.elements() * sizeof(cufftReal), cudaMemcpyDefault);

	// allocate the rest
	cudaMalloc(reinterpret_cast<void **>(&inpCplx), lhs.elements() * sizeof(cufftComplex));
	cudaMalloc(reinterpret_cast<void **>(&kernCplx), k_padded.elements() * sizeof(cufftComplex));
	cudaMalloc(reinterpret_cast<void **>(&outCplx), out.elements() * sizeof(cufftComplex));

    // CUFFT plan to transform the matrices between real and complex
    cufftHandle inpPlan;
    cufftHandle outPlan;
    cufftPlan2d(&inpPlan, W, H, CUFFT_R2C);  // real to complex
    cufftPlan2d(&outPlan, W, H, CUFFT_C2R);  // complex to real
     
    // Transform signal and filter
    cufftExecR2C(inpPlan, inpReal, inpCplx);
    cufftExecR2C(inpPlan, kernReal, kernCplx);

    conv2dFFTKernel<<<1, H * W>>>(inpCplx, kernCplx, outCplx, W, 5);
	cudaDeviceSynchronize();

    // Transform signal back
    cufftExecC2R(outPlan, outCplx, outReal);

	cufftDestroy(inpPlan);
	cufftDestroy(outPlan);

	cudaFree(inpReal);
	cudaFree(kernReal);

	cudaFree(inpCplx);
	cudaFree(kernCplx);
	cudaFree(outCplx);
}
}  // namespace cuda
}  // namespace lt