#include "lantern/tensor/accel/rawcpu/CPUBackend.h"

#include "lantern/tensor/accel/rawcpu/CPUTensor.h"
#include "lantern/tensor/accel/RuntimeCheck.h"

#include <memory>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <limits>

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
    checkMatmulOrThrow(lhs, rhs);

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
    checkConv2dOrThrow(lhs, f, b);

    const dim_t N = lhs.shape()[0], C = lhs.shape()[1], H = lhs.shape()[2], W = lhs.shape()[3],
                 OUT_C = f.shape()[0], K_HW = f.shape()[2];
    const dim_t H_NEW = H - K_HW + 1, W_NEW = W - K_HW + 1;

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

    if (!b.isEmpty() && b.shape().ndim() == 1) {
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

Tensor CPUBackend::add(const Tensor& lhs, const Tensor& rhs) {
    checkAddOrThrow(lhs, rhs);

    unsigned long long M = lhs.shape()[0], N = lhs.shape()[1];

    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(
            Shape{M, N})
    );

    for (long long i = 0; i < M; ++i) {
        for (long long j = 0; j < N; ++j) {
            get(ret, {i, j}) = get(lhs, {i, j}) + get(rhs, {i, j});
        }
    }

    return ret;
}

Tensor CPUBackend::max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    checkMaxPoolOrThrow(lhs, k_sh);

    const dim_t N = lhs.shape()[0], C = lhs.shape()[1], H = lhs.shape()[2], W = lhs.shape()[3];
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
                    float tmp = -std::numeric_limits<float>::max();
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


// TODO: tensor values has not to be of type float (std::max(x, 0.0f))
Tensor CPUBackend::relu(const Tensor& lhs) {
    checkReluOrThrow(lhs);

    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(lhs.shape())
    );

    if (lhs.shape().ndim() == 2) {
        unsigned long long M = lhs.shape()[0], N = lhs.shape()[1];
        for (long long i = 0; i < M; i++) {
            for (long long j = 0; j < N; j++) {
                get(ret, {i, j}) = std::max(get(lhs, {i, j}), 0.0f);
            }
        }
    } else {
        unsigned long long M = lhs.shape()[0], N = lhs.shape()[1], P = lhs.shape()[2], Q = lhs.shape()[3];
        for (long long i = 0; i < M; i++) {
            for (long long j = 0; j < N; j++) {
                for (long long k = 0; k < P; k++) {
                    for (long long l = 0; l < Q; l++) {
                        get(ret, {i, j, k, l}) = std::max(get(lhs, {i, j, k, l}), 0.0f);
                    }
                }
            }
        }
    }

    return ret;
}

Tensor CPUBackend::softmax(const Tensor& lhs) {
    checkSoftmaxOrThrow(lhs);

    unsigned long long M = lhs.shape()[0], N = lhs.shape()[1];

    // get zero initialized result tensor
    auto ptr = lhs.getGate<lt::CPUTensor>().data();
    Tensor ret(
        Tensor::zeros<std::remove_reference_t<decltype(ptr)>::element_type>(lhs.shape())
    );

    float tmp = -std::numeric_limits<float>::max();

    for (long long i = 0; i < M; i++) {
        for (long long j = 0; j < N; j++) {
            if (get(lhs, {i, j}) > tmp) {
                tmp = get(lhs, {i, j});
            }
        }
    }

    float sum = 0;
    for (long long i = 0; i < M; i++) {
        for (long long j = 0; j < N; j++) {
            sum += std::exp(get(lhs, {i, j}) - tmp);
        }
    }

    float offset = tmp + std::log(sum);
    for (long long i = 0; i < M; i++) {
        for (long long j = 0; j < N; j++) {
            get(ret, {i, j}) = std::exp(get(lhs, {i, j}) - offset);
        }
    }

    return ret;
}

}  // namespace lt