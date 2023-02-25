#pragma once

#include "include/lantern.h"
#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>

class Graph {

    std::string modelPath;

    void loadImage(lt::Tensor &input, std::string imagePath);
    void getWeights(onnx::ModelProto &model, std::map<std::string, lt::Tensor*> &collector);
    void getInput(onnx::ModelProto &model, std::map<std::string, lt::Tensor*> &collector);

    public:
        Graph(std::string mdlPath) : modelPath(mdlPath) {};

        void executeGraph(std::string imagePath);
        
};
