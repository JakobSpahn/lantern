#include "include/lantern.h"
#include "lantern/inference/Graph.h"

#include <iostream>

int main() {
    lt::manage::setDefaultGate<lt::CPUTensor>();
    std::cout << lt::manage::getDefaultGate() << std::endl;
    Graph g("../model/model.onnx");
    g.executeGraph("../mnist/testSample/img_1.jpg");

    lt::manage::setDefaultGate<lt::CUDATensor>();
    
    lt::Tensor x(lt::Tensor::fromVector(
        std::vector<float>({-6790.05, -6626.31, 6157.88, -6775.27, -3096.04, -9614.22, -7258.32, -5817.05, -6412.74, -7082.28}), 
        lt::Shape{1,10}));

    auto res = lt::softmax(x);

    std::cout << res.toString() << std::endl;


}