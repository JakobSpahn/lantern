#include "lantern/tensor/accel/rawcpu/CPUBackend.h"
#include "lantern/tensor/accel/rawcpu/CPUTensor.h"

#include <memory>
#include <numeric>
#include <stdexcept>

namespace lt {

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

    return t.getGate<CPUTensor>().data()[n];
}

CPUBackend& CPUBackend::getInstance() {
    static CPUBackend backend;
    return backend;
}

Tensor CPUBackend::reshape(const Tensor& lhs, const Shape& sh) {
    assert(lhs.shape().elements() == sh.elements());

    return Tensor(std::make_unique<CPUTensor>(
        lhs.getGate<CPUTensor>().data(), 
        sh)); // shallow copy with different shape
}

Tensor CPUBackend::transpose(const Tensor& lhs, const Shape& sh) {
    // how do we do this actually?
    assert("not implemented" && false);   
}

Tensor CPUBackend::matmul(const Tensor& lhs, const Tensor& rhs) {
    assert(lhs.shape().elements() > 0 && rhs.shape().elements() > 0);
    if (lhs.shape().ndim() != rhs.shape().ndim()) {
        throw std::invalid_argument("Number of dimensions don't match.");
    }
    if (lhs.shape().ndim() != 2) {
        throw std::invalid_argument("Tensors need to be 2D");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("Shapes don't match");
    }

    unsigned long long M = lhs.shape()[0], N = rhs.shape()[0], P = rhs.shape()[1];

    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(
            Shape{lhs.shape()[0], rhs.shape()[1]})
        );
    
     // actual compute
    // #pragma omp parallel for collapse(3)
    for (long long i = 0; i < M; ++i) {
        for (long long j = 0; j < P; ++j) {
            for (long long k = 0; k < N; ++k) {
                get(ret,{i, j}) += get(lhs,{i, k}) * get(rhs,{k, j});
            }
        }
    }

    return ret;
}

Tensor CPUBackend::conv2d(const Tensor& lhs, const Tensor& f, const Tensor& b) {
    assert(lhs.shape().elements() > 0 && f.shape().elements() > 0);
    assert(lhs.shape().ndim() == f.shape().ndim() && 
        "Input tensor and weight tensor have different dimensions");
    assert(lhs.shape().ndim() == 4 && 
        "Only 4D Tensors accepted, reshape if possible");
    if (b.shape().elements() > 0) assert(b.shape().ndim() == 1 && "if available, bias must be 1D");

    const dim_t N = lhs.shape()[0], C = lhs.shape()[1], H = lhs.shape()[2], W = lhs.shape()[3],
                 OUT_C = f.shape()[0], K_HW = f.shape()[2];
    const dim_t H_NEW = H - K_HW + 1, W_NEW = W - K_HW + 1;

    assert(C == f.shape()[1] && "Input channels don't match with weight tensor");
    assert(K_HW == f.shape()[3] && "Kernel height and width don't match");
    if (b.shape().elements() > 0) assert(OUT_C == b.shape()[0] && "Bias shape doesn't match output channels");

    // zero initialize tensor with new shape (N,OUT_C,H_NEW,W_NEW)
    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(
            Shape{N, OUT_C, H_NEW, W_NEW})
        );

    // convolve
    // #pragma omp parallel for collapse(7)
    for (dim_t batch = 0; batch < N; ++batch) {
        for (dim_t c_out = 0; c_out < OUT_C; ++c_out) {
            for (dim_t c_in = 0; c_in < C; ++c_in) {
                for (dim_t j = 0; j < H_NEW; ++j) {
                    for (dim_t k = 0; k < W_NEW; ++k) {
                        for (dim_t l = 0; l < K_HW; ++l) {
                            for (dim_t m = 0; m < K_HW; ++m) {
                                get(ret, {batch, c_out, j, k}) +=
                                    get(lhs, {batch, c_in, j + l, k + m}) *
                                    get(f, {c_out, c_in, l, m});
                            }
                        }
                    }
                }
            }
        }
    }

    if (b.shape().elements() > 0 && b.shape().ndim() == 1) {
        // add bias
        for (dim_t batch = 0; batch < N; ++batch) {
            for (dim_t c_out = 0; c_out < OUT_C; ++c_out) {
                for (dim_t j = 0; j < H_NEW; ++j) {
                    for (dim_t k = 0; k < W_NEW; ++k) {
                        get(ret, {batch, c_out, j, k}) += get(b, {c_out});
                    }
                }
            }
        }
    }

    return ret;
}

Tensor CPUBackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    assert(lhs.shape().elements() > 0);
    assert(k_sh.ndim() == 2 && "Kernel not 2D");
    assert(k_sh[0] == k_sh[1] &&
            "Kernel height not equal to width");

    const dim_t N = lhs.shape()[0], C = lhs.shape()[1], H = lhs.shape()[2], W = lhs.shape()[3];
    assert((H % k_sh[0]) == 0 && (W % k_sh[0]) == 0 &&
            "Kernel shape and input tensor width/height missmatch");
    const dim_t H_NEW = H / k_sh[0], W_NEW = W / k_sh[0];
    const dim_t stride = k_sh[0];

    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(
            Shape{N, C, H_NEW, W_NEW})
        );

    for (dim_t batch = 0; batch < N; ++batch) {
        for (dim_t c = 0; c < C; ++c) {
            for (dim_t j = 0; j < H_NEW; ++j) {
                for (dim_t k = 0; k < W_NEW; ++k) {
                    float tmp = -INFINITY;
                    for (dim_t l = j * stride; l < (j + 1) * stride; ++l) {
                        for (dim_t m = k * stride; m < (k + 1) * stride; ++m) {
                            const float ref = get(lhs, {batch, c, l, m});
                            if (ref > tmp) {
                                tmp = ref;
                            }
                        }
                    }
                    get(ret, {batch, c, j, k}) = tmp;
                }
            }
        }
    }

    return ret;
}


}  // namespace lt