#include "tensor.h"

#include <algorithm>

#define assertm(exp, msg) assert(((void)msg, exp))

/*
If just given the shape, the tensor will zero initialize
*/
Tensor::Tensor(shape_t shape) : shape(shape) {
    unsigned int n = t::mul_shape_elements(shape);

    dat = new float[n]();  // gets zero initialized
    empty = false;
}

Tensor::Tensor(const float* inp, unsigned int n, const shape_t& shape)
    : shape(shape) {
    assertm(n == t::mul_shape_elements(shape),
            "Number of input elements don't match the specified shape");
    assertm(inp, "Input data is empty!");

    dat = new float[n];
    dat = reinterpret_cast<float*>(std::memcpy(dat, inp, n));
    empty = false;
}

Tensor::Tensor(const Tensor& rhs) {
    assertm(!rhs.empty, "Tensor to be copied is empty!");

    unsigned int n = t::mul_shape_elements(rhs.shape);

    dat = new float[n];
    dat = reinterpret_cast<float*>(std::memcpy(dat, rhs.dat, n));

    shape = rhs.shape;
    empty = rhs.empty;
}

Tensor::Tensor(const std::vector<float>& inp, shape_t shape) : shape(shape) {
    unsigned int n = inp.size();
    assertm(n == t::mul_shape_elements(shape),
            "Number of input elements don't match the specified shape");

    dat = new float[n];
    std::copy(inp.begin(), inp.end(), dat);
    empty = false;
}

Tensor& Tensor::operator=(const Tensor& rhs) {
    assertm(!rhs.empty, "Tensor to be copied is empty!");

    unsigned int n = t::mul_shape_elements(rhs.shape);

    if (dat) {
        delete dat;
    }
    dat = new float[n];
    dat = reinterpret_cast<float*>(std::memcpy(dat, rhs.dat, n));

    shape = rhs.shape;

    empty = rhs.empty;

    return *this;
}

/*
Discards current shape
*/
Tensor& Tensor::operator=(const std::vector<float>& inp) {
    unsigned int n = inp.size();

    if (dat) {
        delete dat;
    }
    dat = new float[n];
    dat = std::copy(inp.begin(), inp.end(), dat);

    shape = shape_t{n};

    empty = false;

    return *this;
}

float& Tensor::operator[](const shape_t& access) {
    // assertm(!empty, "Tensor is empty"); // --> not sure if this is wanted
    assertm(shape.size() == access.size(),
            "Number of indices doesn't match shape of tensor");

    size_t max_dims = access.size();
    unsigned int n = 0;

    for (size_t i = 0; i < access.size(); ++i) {
        assertm(access[i] < shape[i], "Access index out of bounds");

        if (i < (max_dims - 1)) {
            n += access[i] *
                 t::mul_shape_elements(shape.begin() + i + 1, shape.end());
        } else {
            n += access[i];
        }
    }
    return dat[n];
}

float Tensor::operator[](const shape_t& access) const {
    // assertm(!empty, "Tensor is empty"); // --> not sure if this is wanted
    assertm(shape.size() == access.size(),
            "Number of indices doesn't match shape of tensor");

    size_t max_dims = access.size();
    unsigned int n = 0;

    for (size_t i = 0; i < access.size(); ++i) {
        assertm(access[i] < shape[i], "Access index out of bounds");

        shape_t shp(shape);
        if (i < (max_dims - 1)) {
            n += access[i] *
                 t::mul_shape_elements(shp.begin() + i + 1, shp.end());
        } else {
            n += access[i];
        }
    }
    return dat[n];
}

std::ostream& Tensor::print_shape(std::ostream& os) {
    auto it = shape.begin();
    os << "(" << *(it++);
    while (it != shape.end()) {
        os << ", " << *(it++);
    }
    os << ")";
    return os;
}
