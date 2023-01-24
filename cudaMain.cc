// clang++ --std=c++20 lantern/tensor/*.cc lantern/tensor/accel/cuda/*.cc -I . -o cuda
#include "include/lantern.h"
#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include <iostream>

int main() {
    lt::manage::setDefaultGate<lt::CUDATensor>();

    auto x(lt::Tensor::randn<float>({1, 1, 28, 28})),
            y(lt::Tensor::randn<float>({1, 1, 5, 5})),
            b(lt::Tensor::randn<float>({1}));


    lt::CUDABackend::getInstance().conv_fft = true;
    auto res = lt::conv2d(x, y, b);
    std::cout << "res: " << res << std::endl;
    lt::CUDABackend::getInstance().conv_fft = false;
    auto res2 = lt::conv2d(x, y, b);
    std::cout << "res2: " << res2 << std::endl;


    return 0;
}