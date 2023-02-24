#include "lantern/tensor/accel/RuntimeCheck.h"

namespace lt {

#define EMPTY "Tensors can't be empty"
#define SHP "Shapes don't match"

void checkAddOrThrow(const Tensor& a, const Tensor& b) {
    if (a.isEmpty() || b.isEmpty()) {
        throw std::invalid_argument(EMPTY);
    }

    if (a.shape().ndim() != b.shape().ndim()) {
        throw std::invalid_argument(SHP);
    }

    if (a.shape().ndim() != 2) {
        throw std::invalid_argument("Tensors need to be 2D");
    }

    if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1]) {
        throw std::invalid_argument(SHP);
    }
}

void checkMatmulOrThrow(const Tensor& a, const Tensor& b) {
    if (a.isEmpty() || b.isEmpty()) {
        throw std::invalid_argument(EMPTY);
    }
    if (a.shape().ndim() != b.shape().ndim()) {
        throw std::invalid_argument(SHP);
    }
    if (a.shape().ndim() != 2) {
        throw std::invalid_argument("Tensors need to be 2D");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(SHP);
    }
}

void checkConv2dOrThrow(const Tensor& a, const Tensor& f, const Tensor& b) {
    if (a.isEmpty() || f.isEmpty()) {
        throw std::invalid_argument(EMPTY);
    }
    if (a.shape().ndim() != f.shape().ndim()) {
        throw std::invalid_argument(SHP);
    }
    if (a.shape().ndim() != 4) {
        throw std::invalid_argument("Input tensor needs to be 4D");
    }
    if (!b.isEmpty() && b.shape().ndim() != 1) {
        throw std::invalid_argument("Bias must be 1D");
    }
    const dim_t C{a.shape()[1]},
                 OUT_C{f.shape()[0]},
                 K_HW{f.shape()[2]};
    if (C != f.shape()[1]) {
        throw std::invalid_argument("Input channels don't match with weight tensor");
    }
    if (K_HW != f.shape()[3]) {
        throw std::invalid_argument("Kernel height and width don't match");
    }
    if (!b.isEmpty() && OUT_C != b.shape()[0]) {
        throw std::invalid_argument("Bias shape doesn't match output channels");
    }
}

void checkMaxPoolOrThrow(const Tensor& a, const Shape& k_sh) {
    if (a.isEmpty()) {
        throw std::invalid_argument(EMPTY);
    }
    if (k_sh.ndim() != 2) {
        throw std::invalid_argument("Kernel not 2D");
    }
    if (k_sh[0] != k_sh[1]) {
        throw std::invalid_argument("Kernel height not equal to width");
    }
    const dim_t H = a.shape()[2], W = a.shape()[3];
    if ((H % k_sh[0]) != 0 && (W % k_sh[0]) != 0) {
        throw std::invalid_argument("Kernel shape and input tensor width/height missmatch");
    }
}
}  // namespace lt