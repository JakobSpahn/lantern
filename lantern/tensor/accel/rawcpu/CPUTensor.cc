#include "lantern/tensor/accel/rawcpu/CPUTensor.h"
#include "lantern/tensor/accel/rawcpu/CPUBackend.h"
#include "lantern/tensor/Types.h"

#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iterator>
#include <cassert>

namespace lt {

CPUTensor::CPUTensor(
        const void* dat, 
        const Shape& s, 
        const lt::dtype dt) 
        : arr_(new data_t[s.elements()]),
        sh(s),
        dt(dt) {
            const data_t* data = reinterpret_cast<const data_t*>(dat);
            std::copy(data, data + s.elements(), arr_.get());
        } 

std::unique_ptr<TensorGate> CPUTensor::clone() {
    return std::make_unique<CPUTensor>(arr_.get(), sh, dt);
} 

CPUTensor::CPUTensor(
        std::shared_ptr<data_t[]> dat,
        const Shape& s) 
        : arr_(dat), sh(s) {}

void CPUTensor::assign(const Tensor& t) {
    // TODO
    assert("not implemented" && false);
}

Tensor CPUTensor::copy() {
    assert("not implemented" && false);
}

Tensor CPUTensor::shallowCopy() {
    assert("not implemented" && false);
}

TensorBackend& CPUTensor::backend() const {
    return CPUBackend::getInstance();
}

const Shape& CPUTensor::shape() const {
    return sh;
} 

Tensor CPUTensor::index(const Shape& idx) const {
    using dim_t = long long int;

    dim_t begin = 0;
    // Shape new_sh(sh);

    if (idx.ndim() > sh.ndim()) {
        throw std::invalid_argument("Index has too many dimensions.");
    }

    Shape new_sh((idx.ndim() == sh.ndim() ? std::vector<dim_t>{1} :
                std::vector<dim_t>(sh.get().cbegin() + idx.ndim(), sh.get().cend())));

    for (size_t i = 0; i < idx.ndim(); ++i) {
        if(idx.dim(i) > (sh.dim(i) - 1)) {
            throw std::invalid_argument("Index out of bounds");
        }

        Shape shp(sh);
        begin += idx.dim(i) * (i < (shp.ndim() - 1) 
            ? std::accumulate(
                shp.get().cbegin() + i + 1, 
                shp.get().cend(), 
                1, 
                std::multiplies<dim_t>()) 
            : 1);
    }

    return Tensor(
        std::make_unique<CPUTensor>(
            std::shared_ptr<data_t[]>(arr_, arr_.get() + begin),
            new_sh));
}

std::string CPUTensor::toString() const {
    std::stringstream ss;
    ss << "tensor(";
    
    auto cpy(sh.get());
    auto ptr(cpy.begin());
    auto data_ptr = arr_.get();

    ss << "[";

    while (true) {
        if (*ptr <= 0) {
            ss << "]" << (*(ptr-1) == 0 || ptr == cpy.begin() ? "" : ", ");

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

std::shared_ptr<data_t[]> CPUTensor::data() {
    return arr_;
}

const std::shared_ptr<data_t[]> CPUTensor::data() const {
    return arr_;
}

void CPUTensor::buff(void** out) const {
    *out = data().get();
}

}  // namespace lt

