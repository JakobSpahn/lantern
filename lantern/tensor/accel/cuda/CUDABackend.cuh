#pragma once

#include <cassert>
#include "lantern/tensor/TensorBackend.h"
#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/accel/RuntimeCheck.h"

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

namespace lt {
#define BLOCK_DIM 32
__global__ void mm_kernel(data_t const* mat_1, data_t const* mat_2, data_t* mat_3, size_t m,
						  size_t n, size_t p);
__global__ void batched_channeled_conv2d_hw_kernel(data_t const* inp, data_t const* kernel, data_t* outp,
												   dim_t b_s, dim_t OUT_C, dim_t IN_C,
												   dim_t H_OLD, dim_t W_OLD,
												   dim_t H_NEW, dim_t W_NEW, dim_t K_HW);
__global__ void add_kernel(data_t const* inp_1, data_t const* inp_2, data_t* outp, dim_t H, dim_t W);
__global__ void relu_kernel(data_t const* inp, data_t* outp, dim_t n);
__global__ void softmax_kernel(data_t* inp, data_t* outp, dim_t n);
__global__ void max_pool2d_kernel(data_t* inp, data_t* outp, dim_t input_rows, dim_t input_cols, dim_t kernel_rows, dim_t kernel_cols, dim_t max_val);

template <class T>
class CUDABackend : public lt::TensorBackend {
 public:
    CUDABackend() = default;
    ~CUDABackend() = default;

    static CUDABackend& getInstance(); 

    /******************** ML Operators ********************/
    Tensor reshape(const Tensor& lhs, const Shape& sh);
    Tensor transpose(const Tensor& lhs, const Shape& sh);
    Tensor matmul(const Tensor& lhs, const Tensor& rhs);
    Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b);
    Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh);
    Tensor add(const Tensor& lhs, const Tensor& rhs);
    Tensor relu(const Tensor& lhs);
    Tensor softmax(const Tensor& lhs);
};

template <class T>
CUDABackend<T>& CUDABackend<T>::getInstance() {
	static CUDABackend instance;
	return instance;
}

/******************** ML Operators ********************/
template <class T>
Tensor CUDABackend<T>::reshape(const Tensor& lhs, const Shape& sh) {
	assert(lhs.shape().elements() == sh.elements());

	return Tensor(std::make_unique<CUDATensor>(
		lhs.getGate<CUDATensor>().data(),
		sh
	)); // shallow copy with different shape
}

template <class T>
Tensor CUDABackend<T>::transpose(const Tensor& lhs, const Shape& sh) {
	assert(0 && "not implemented");
	return Tensor();
}

template <class T>
Tensor CUDABackend<T>::matmul(const Tensor& lhs, const Tensor& rhs) {
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

template <class T>
Tensor CUDABackend<T>::conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
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

template <class T>
Tensor CUDABackend<T>::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
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

template <class T>
Tensor CUDABackend<T>::add(const Tensor& lhs, const Tensor& rhs) {
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

template <class T>
Tensor CUDABackend<T>::relu(const Tensor& lhs) {
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

template <class T>
Tensor CUDABackend<T>::softmax(const Tensor& lhs) {
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

}  // namespace lt