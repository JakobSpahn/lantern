#include "lantern/tensor/accel/cuda/CUDATensor.h"

#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include <string>
#include <cassert>
#include <sstream>
#include <memory>

namespace lt {

CUDATensor::~CUDATensor() {
    cudaFree(arr_);
}

CUDATensor::CUDATensor(
    const void* dat, 
    const Shape& s, 
    const lt::dtype dt) 
    : sh(s) {
    cudaMallocManaged(&arr_, s.elements() * sizeof(data_t));
    cudaMemcpy(arr_, dat, s.elements() * sizeof(data_t), cudaMemcpyDefault);
    cudaDeviceSynchronize();
}

CUDATensor::CUDATensor(
        data_t* dat,
        const Shape& s) 
        : arr_(dat), sh(s) {}
    
std::unique_ptr<TensorGate> CUDATensor::clone() {
    return std::make_unique<CUDATensor>(arr_, sh, lt::dtype::float32);
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
    return CUDABackend::getInstance();
}

const Shape& CUDATensor::shape() const {
    return sh;
}

Tensor CUDATensor::index(const Shape& sh) const {
    assert(0 && "not implemented");
    return Tensor();
}

std::string CUDATensor::toString() const {
    std::stringstream ss;
    ss << "tensor(";

    auto cpy(sh.get());
    auto ptr(cpy.begin());
    auto data_ptr = arr_;

    ss << "[";

    while (true) {
        if (*ptr <= 0) {
            ss << "]" << (*(ptr - 1) == 0 || ptr == cpy.begin() ? "" : ", ");

            if (ptr == cpy.begin()) {
                break;
            }

            *ptr-- = *(sh.get().cbegin() + std::distance(cpy.begin(), ptr));

        } else if (ptr != cpy.end() - 1) {
            ss << "[";
            (*ptr++)--;

        } else {
            for (size_t i = 0; i < *ptr; ++i) {
                ss << *data_ptr++ << (i == *ptr - 1 ? "" : ", ");
            }

            *ptr = 0;
        }
    }

    ss << ")";
    return ss.str();
}

void CUDATensor::buff(void** out) const {
    *out = arr_;
}

data_t* CUDATensor::data() {
    return arr_;
}

const data_t* CUDATensor::data() const {
    return arr_;
}

}  // namespace lt