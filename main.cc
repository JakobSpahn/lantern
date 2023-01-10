// clang++ --std=c++20 lantern/tensor/*.cc lantern/tensor/accel/rawcpu/*.cc -I . -o rawcpu
#include "include/lantern.h"

#include <vector>
#include <iostream>

int main() {
    lt::manage::setDefaultGate<lt::CPUTensor>();

    lt::Tensor x(lt::Tensor::fromVector(
        std::vector<float>({1,2,3,4,5}), 
        lt::Shape{1,5}));
    lt::Tensor y(lt::Tensor::fromVector(
        std::vector<float>({1,2,3,4,5,6,7,8,9,10}), 
        lt::Shape{5,2}));

    lt::Tensor z(lt::Tensor::zeros<float>(
        lt::Shape{1,65}));
    
    auto res = lt::matmul(x,y);

    float* el = res.buff<float>();
    
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << res << std::endl;
    std::cout << *el << std::endl;
    std::cout << z << std::endl;
}