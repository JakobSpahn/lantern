// clang++ --std=c++20 lantern/tensor/*.cc lantern/tensor/accel/cuda/*.cc -I . -o cuda
#include "include/lantern.h"

#include <iostream>
// #include <onnx/onnx_pb.h>
// #include <onnx/onnx-operators_pb.h>

int main() {

    lt::manage::setDefaultGate<lt::CUDATensor>();

    lt::Tensor x(lt::Tensor::randn<float>(
        lt::Shape{10, 10}
    ));
    lt::Tensor y(lt::Tensor::randn<float>(
        lt::Shape{10, 10}
    ));
    lt::Tensor z(x);


    lt::Tensor result = lt::matmul(x, y);

    float* ptr_x = x.buff<float>();
    float* ptr_y = y.buff<float>();
    float* ptr_z = z.buff<float>();

    *ptr_x = 1;

    std::cout << *ptr_x << std::endl;
    std::cout << *ptr_y << std::endl;
    std::cout << *ptr_z << std::endl;

	std::cout << x << std::endl;
	std::cout << z << std::endl;
	std::cout << y << std::endl;
	std::cout << result << std::endl;

    /*
    // test for memory leak
    while(true) {
        auto tmp(z);
    }
    */

    return 0;
}