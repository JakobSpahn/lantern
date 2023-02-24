#include "Graph.h"

#include <fstream>
#include <vector>
#include <onnx/onnx_pb.h>
#include <onnx/onnx-operators_pb.h>
#include <opencv2/opencv.hpp>

void get_input(onnx::ModelProto &model, std::map<std::string, lt::Tensor*> &collector) 
{
    for(auto& in : model.graph().input())
    {        
        lt::Shape dims;
        if(in.type().has_tensor_type())
        {
            if(in.type().tensor_type().has_shape())
            {
                auto in_dims = in.type().tensor_type().shape().dim();
                for(auto& d : in_dims)
                {   
                    dims.addDim(d.dim_value());
                }
                // Set batch size = 1
                dims[0] = 1;
                lt::Tensor* t = new lt::Tensor(lt::Tensor::zeros<float>(dims));
                t->name = in.name();
                collector[t->name] = t;
            }
        }
    }
}

void get_weights(onnx::ModelProto &model, std::map<std::string, lt::Tensor*> &collector)
{
    for (auto& info : model.graph().initializer()) {
        lt::Shape dims;
        for (auto& dim : info.dims()) {
            dims.addDim(dim);
        }
        if (info.data_type() == onnx::TensorProto_DataType_FLOAT) 
        {
            lt::Tensor *t = new lt::Tensor(
                lt::Tensor::fromBuffer(
                    (float *) info.raw_data().data(),
                    dims
                )
            );
            t->name = info.name();
            collector[t->name] = t;
        } else if (info.data_type() == onnx::TensorProto_DataType_INT64)
        {
            // TODO: Currently we cast to float however integer tensor are available too
            int64_t* raw_data = (int64_t*) info.raw_data().data();
            int n = info.raw_data().size() / sizeof(int64_t);
            float* data = (float*) malloc(n * sizeof(float));
            for (int i = 0; i < n; i++) {
                data[i] = static_cast<float>(abs(raw_data[i]));
            }
            lt::Tensor* t = new lt::Tensor(
                lt::Tensor::fromBuffer(
                    data,
                    dims
                )
            );
            t->name = info.name();
            collector[t->name] = t;
        } else {
            std::cout<<"Type: "<<info.data_type()<<" unknown."<<std::endl;
            abort();
        }     
    }

} 

void load_image(lt::Tensor &input, std::string filepath)
{
    cv::Mat src = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    cv::Mat img;
    cv::copyMakeBorder(src, img, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0);

    std::vector<float> buffer(img.rows*img.cols);
    for (int i = 0; i < img.rows*img.cols; i++)
    {
        buffer[i] = static_cast<float>(img.data[i]);
    }

    input = lt::Tensor::fromVector<float>(buffer, input.shape());
}

void executeGraph(std::string modelPath, std::string imagePath)
{
    onnx::ModelProto model;
    std::ifstream in(modelPath, std::ios_base::binary);
    model.ParseFromIstream(&in);
    in.close();

    lt::manage::setDefaultGate<lt::CPUTensor>();
    std::map<std::string, lt::Tensor*> collector;
    get_input(model, collector);
    get_weights(model, collector);

    lt::Tensor* input = collector[model.graph().input()[0].name()];
    load_image(*input, imagePath);
    lt::Tensor ret;

    for (auto& nd_proto : model.graph().node())
    {
        std::vector<std::string> inp;
        for (auto& in : nd_proto.input()) {
            inp.push_back(in);
        }

        std::string op_name = nd_proto.op_type();
        if (op_name == "Conv")
        {
            ret = lt::conv2d(
                *collector[inp[0]], *collector[inp[1]], *collector[inp[2]]
            );
        } else if (op_name == "Relu")
        {
            //ret = relu(
            //    *collector[inp[0]]
            //);
        } else if (op_name == "MaxPool")
        {
            lt::Shape kernel_shape;
            for (const auto& attr : nd_proto.attribute()) {
                if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); i++) {
                        kernel_shape.addDim(
                            attr.ints(i)
                        );
                    }
                }
            }
            ret = lt::max_pool2d(
                *collector[inp[0]],
                kernel_shape
            );
        } else if (op_name == "MatMul")
        {
            ret = lt::matmul(
                *collector[inp[0]], *collector[inp[1]]
            );
        } else if (op_name == "Add")
        {
            ret = lt::add(
              *collector[inp[0]], *collector[inp[1]]
            );
        } else if (op_name == "Softmax")
        {
            //ret = softmax(
            //    *collector[inp[0]]
            //);
        } else if (op_name == "Reshape")
        {   
            lt::Shape new_shape;
            float* buffer = collector[inp[1]]->buff<float>();
            for (int i = 0; i < collector[inp[1]]->elements(); i++) {
                new_shape.addDim(static_cast<int64_t>(buffer[i]));
            }
            ret = lt::reshape(
                *collector[inp[0]], new_shape
            );
            
        }
        collector[nd_proto.output()[0]] = &ret;
    }
}
