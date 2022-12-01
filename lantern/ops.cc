#include <algorithm>
#include <cmath>
#include <iostream>

#include "tensor.h"

// TODO: check later if methods can be made void or need to return copy

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    assertm(t::mul_shape_elements(shape) == t::mul_shape_elements(new_shape),
            "Shapes don't match");
    shape = new_shape;
}

void Tensor::permute(const shape_t& permutation) {}

void Tensor::matmul(const Tensor& w) {
    assert(!empty && !w.empty);
    assertm(shape.size() == w.shape.size(), "Dimensions don't match");
    assertm(shape.size() == 2, "Tensors need to be 2D");
    assertm(
        shape[1] == w.shape[0],
        "Input tensor is of shape MxN but weight tensor is not of shape NxP");

    size_t M = shape[0], N = w.shape[0], P = w.shape[1];

    // zero initialize tensor with shape
    Tensor ret{{shape[0], w.shape[1]}};  // output tensor is of shape MxP

    // actual compute
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < P; ++j) {
            for (size_t k = 0; k < N; ++k) {
                ret[{i, j}] += (*this)[{i, k}] * w[{k, j}];
            }
        }
    }

    *this = ret;
}

/*
(*this): input tensor of shape (N,C,H,W)
w: tensor containing the filter, is of shape (OUT_C, C, K_HW, K_HW)
b: bias tensors shape (OUT_C)
padding: for now only valid padding available
*/
void Tensor::conv2d(const Tensor& w, const Tensor& b,
                    const std::string& padding) {
    assert(!empty && !w.empty);
    assertm(shape.size() == w.shape.size(),
            "Input tensor and weight tensor have different dimensions");
    assertm(shape.size() == 4, "Only 4D Tensors accepted, reshape if possible");
    if (!b.empty) assertm(b.shape.size() <= 1, "if available, bias must be 1D");

    const size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3],
                 OUT_C = w.shape[0], K_HW = w.shape[2];
    const size_t H_NEW = H - K_HW + 1, W_NEW = W - K_HW + 1;

    assertm(C == w.shape[1], "Input channels don't match with weight tensor");
    assertm(K_HW == w.shape[3], "Kernel height and width don't match");
    if (!b.empty) assertm(OUT_C == b.shape[0], "Bias shape doesn't match output channels");

    // zero initialize tensor with new shape (N,OUT_C,H_NEW,W_NEW)
    Tensor ret{{N, OUT_C, H_NEW, W_NEW}};

    // convolve
    for (size_t batch = 0; batch < N; ++batch) {
        for (size_t c_out = 0; c_out < OUT_C; ++c_out) {
            for (size_t c_in = 0; c_in < C; ++c_in) {
                for (size_t j = 0; j < H_NEW; ++j) {
                    for (size_t k = 0; k < W_NEW; ++k) {
                        for (size_t l = 0; l < K_HW; ++l) {
                            for (size_t m = 0; m < K_HW; ++m) {
                                ret[{batch, c_out, j, k}] +=
                                    (*this)[{batch, c_in, j + l, k + m}] *
                                    w[{c_out, c_in, l, m}];
                            }
                        }
                    }
                }
            }
        }
    }

    if (!b.empty && b.shape.size() == 1) {
        // add bias
        for (size_t batch = 0; batch < N; ++batch) {
            for (size_t c_out = 0; c_out < OUT_C; ++c_out) {
                for (size_t j = 0; j < H_NEW; ++j) {
                    for (size_t k = 0; k < W_NEW; ++k) {
                        ret[{batch, c_out, j, k}] += b[{c_out}];
                    }
                }
            }
        }
    }

    (*this) = ret;
}

void Tensor::max_pool(const shape_t& kernel_shape) {
    assert(!empty);
    assertm(kernel_shape.size() == 2, "Kernel not 2D");
    assertm(kernel_shape[0] == kernel_shape[1],
            "Kernel height not equal to width");

    const size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    assertm((H % kernel_shape[0]) == 0 && (W % kernel_shape[0]) == 0,
            "Kernel shape and input tensor width/height missmatch");
    const size_t H_NEW = H / kernel_shape[0], W_NEW = W / kernel_shape[0];
    const size_t stride = kernel_shape[0];

    Tensor ret{{N, C, H_NEW, W_NEW}};

    for (size_t batch = 0; batch < N; ++batch) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t j = 0; j < H_NEW; ++j) {
                for (size_t k = 0; k < W_NEW; ++k) {
                    float tmp = -INFINITY;
                    for (size_t l = j * stride; l < (j + 1) * stride; ++l) {
                        for (size_t m = k * stride; m < (k + 1) * stride; ++m) {
                            const float ref = (*this)[{batch, c, l, m}];
                            if (ref > tmp) {
                                tmp = ref;
                            }
                        }
                    }
                    ret[{batch, c, j, k}] = tmp;
                }
            }
        }
    }
    (*this) = ret;
}

void Tensor::add(const Tensor& b) {
    assert(!empty && !b.empty);
    assertm(b.shape.size() == 1, "Bias has wrong shape");
    assertm(shape.size() == 2, "This operation is meant for 2D input tensor");
    assertm(b.shape[0] == shape[1], "Input tensor has wrong shape");

    const size_t H = shape[0], W = shape[3];

    // add bias
    for (size_t j = 0; j < H; ++j) {
        for (size_t k = 0; k < W; ++k) {
            (*this)[{j, k}] += b[{k}];
        }
    }
}

void Tensor::relu() {
    assert(!empty);

    const unsigned int n = t::mul_shape_elements(shape);
    for (unsigned int j = 0; j < n; ++j) {
        dat[j] = std::max(dat[j], static_cast<float>(0.0));
    }
}

void Tensor::softmax() {
    assert(!empty);

    float m = -INFINITY;
    const unsigned int n = t::mul_shape_elements(shape);

    for (unsigned int j = 0; j < n; j++) {
        if (dat[j] > m) {
            m = dat[j];
        }
    }

    float sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += std::expf(dat[i] - m);
    }

    float offset = m + std::logf(sum);
    for (size_t i = 0; i < n; i++) {
        dat[i] = std::expf(dat[i] - offset);
    }
}
