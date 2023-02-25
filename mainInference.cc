#include "include/lantern.h"
#include "lantern/inference/Graph.h"

#include <iostream>

int main() {
    lt::manage::setDefaultGate<lt::CPUTensor>();
    Graph g("../model/model.onnx");
    g.executeGraph("../mnist/testSample/img_1.jpg");
    //executeGraph("../model/model.onnx","../mnist/testSample/img_1.jpg");
    /*
    lt::manage::setDefaultGate<lt::CPUTensor>();

    lt::Tensor x(lt::Tensor::fromVector(
        std::vector<float>({1,2,3,4,5,6,7,8,9,10}), 
        lt::Shape{5,2}));

    lt::Tensor y(lt::Tensor::fromVector(
        std::vector<float>({1,2,3,4,5,6,7,8,9,10}), 
        lt::Shape{5,2}));

    std::cout << lt::add(x, y) << std::endl;
    */
}