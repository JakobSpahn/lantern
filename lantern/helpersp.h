#pragma once

#include <vector>

#include "libnpy/include/npy.hpp"
#include "tensor.h"

namespace p {

Tensor load_npy(const std::string& path, shape_t shape) {
    std::vector<unsigned long> shape_np{};
    bool fortran_order;
    std::vector<double> data;

    npy::LoadArrayFromNumpy(path, shape_np, fortran_order, data);
    std::vector<float> fdata(data.begin(), data.end());  // convert to float
    return Tensor{fdata,shape};
}

}  // namespace p