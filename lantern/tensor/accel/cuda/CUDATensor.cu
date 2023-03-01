#include "lantern/tensor/accel/cuda/CUDATensor.h"

#include "lantern/tensor/accel/cuda/CUDABackend.cuh"

#include "lantern/tensor/Types.h"

#include <string>
#include <cassert>
#include <sstream>
#include <memory>
#include <iostream>

#include "lantern/tensor/Types.h"

namespace lt {

CUDATensor::~CUDATensor() {
    cudaFree(arr_);
}

CUDATensor::CUDATensor(
    const void* dat, 
    const Shape& s, 
    const lt::dtype dt) 
    : sh(s), dt(dt) {
    cudaMallocManaged(&arr_, s.elements() * getTypeSize(dt));
    cudaMemcpy(arr_, dat, s.elements() * getTypeSize(dt), cudaMemcpyDefault);
    cudaDeviceSynchronize();
}

CUDATensor::CUDATensor(
        void* dat,
        const Shape& s) 
        : arr_(dat), sh(s) {}
    
std::unique_ptr<TensorGate> CUDATensor::clone() {
    return std::make_unique<CUDATensor>(arr_, sh, dt);
}

void CUDATensor::assign(const Tensor& t) {
    assert(0 && "not implemented");
}

Tensor CUDATensor::copy() {
    assert(0 && "not implemented");
    return Tensor();
}

Tensor CUDATensor::shallowCopy() {
    assert(0 && "not implemented");
    return Tensor();
}

TensorBackend& CUDATensor::backend() const {
	std::cout << getTypeName(dt) << std::endl;
	switch(dt) {
		case dtype::float32: return CUDABackend<float>::getInstance();
		default:
			throw std::invalid_argument("dtype not implemented");
			break;
	}
	// should never reach
    return CUDABackend<float>::getInstance();
}

const Shape& CUDATensor::shape() const {
    return sh;
}

Tensor CUDATensor::index(const Shape& sh) const {
    assert(0 && "not implemented");
    return Tensor();
}

std::string CUDATensor::toString() const {
    assert(0 && "deprecated");
	return "";
}

void CUDATensor::buff(void** out) const {
    *out = arr_;
}
}  // namespace lt