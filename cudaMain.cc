// clang++ --std=c++20 lantern/tensor/*.cc lantern/tensor/accel/cuda/*.cc -I . -o cuda
#include "include/lantern.h"

#include <iostream>
#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>

int main() {
    lt::manage::setDefaultGate<lt::CUDATensor>();

    lt::Tensor x(lt::Tensor::zeros<float>(
        lt::Shape{1000000}
    ));
    lt::Tensor y(lt::Tensor::zeros<float>(
        lt::Shape{1000000}
    ));
    lt::Tensor z(x);

    float* ptr_x = x.buff<float>();
    float* ptr_y = y.buff<float>();
    float* ptr_z = z.buff<float>();

    *ptr_x = 1;

    std::cout << *ptr_x << std::endl;
    std::cout << *ptr_y << std::endl;
    std::cout << *ptr_z << std::endl;

    /*
    // test for memory leak
    while(true) {
        auto tmp(z);
    }
    */

    return 0;
}