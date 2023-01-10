#include <chrono>
#include <iostream>

#include "lantern/tensor/Tensor.h"
#include "lantern/tensor/accel/rawcpu/CPUTensor.h"
#include "lantern/tensor/accel/cuda/CUDATensor.h"
#include "lantern/tensor/TensorBackend.h"
#include "lantern/tensor/Factory.h"

#include "lantern/tensor/Shape.h"

void update_out_param(const lt::Tensor& x, float** out_data, unsigned int* out_data_n, int** out_shape, unsigned int* out_shape_n) {
    float* new_data = new float[x.elements()];
    std::copy(x.buff<float>(), x.buff<float>() + x.elements(), new_data);

    int* new_shape = new int[x.shape().ndim()];
    std::copy(x.shape().get().begin(), x.shape().get().end(), new_shape);

    *out_data = new_data;
    *out_data_n = x.elements();
    *out_shape = new_shape;
    *out_shape_n = x.shape().ndim();
}

/*
void matmul(lt::Tensor& a, const lt::Tensor& b) { a.matmul(b); }
void conv2d(Tensor& a, const Tensor& b, const Tensor& c) { a.conv2d(b, c); }
void max_pool2d(Tensor& x, const shape_t& kernel_shape) { x.max_pool(kernel_shape); }
*/

extern "C" {
double matmul(const float* a_data, const unsigned int a_data_n,
              const int* a_shape, const unsigned int a_shape_n,
              const float* b_data, const unsigned int b_data_n,
              const int* b_shape, const unsigned int b_shape_n,
              float** out_data, unsigned int* out_data_n,
              int** out_shape, unsigned int* out_shape_n) {
    lt::manage::setDefaultGate<lt::CUDATensor>();

    std::vector<lt::dim_t> a_sh{a_shape, a_shape + a_shape_n},
                            b_sh{b_shape, b_shape + b_shape_n};

    lt::Tensor a(lt::Tensor::fromBuffer(a_data, lt::Shape(a_sh)));
    const lt::Tensor b{lt::Tensor::fromBuffer(b_data, lt::Shape(b_sh))};

    auto st = std::chrono::steady_clock::now();
    auto res = lt::matmul(a, b);
    auto ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(res, out_data, out_data_n, out_shape, out_shape_n);

    return dsec.count();
}

double conv2d(const float* a_data, const unsigned int a_data_n,
              const int* a_shape, const unsigned int a_shape_n,
              const float* b_data, const unsigned int b_data_n,
              const int* b_shape, const unsigned int b_shape_n,
              const float* c_data, const unsigned int c_data_n,
              const int* c_shape, const unsigned int c_shape_n, const bool use_c, 
              float** out_data, unsigned int* out_data_n,
              int** out_shape, unsigned int* out_shape_n) {
    lt::manage::setDefaultGate<lt::CUDATensor>();
    
    std::vector<lt::dim_t> a_sh{a_shape, a_shape + a_shape_n}, 
                            b_sh{b_shape, b_shape + b_shape_n},
                            c_sh{c_shape, c_shape + c_shape_n};

    lt::Tensor a(lt::Tensor::fromBuffer(a_data, lt::Shape(a_sh)));
    const lt::Tensor b{lt::Tensor::fromBuffer(b_data, lt::Shape(b_sh))}; 
    lt::Tensor c;

    if (use_c) { 
        c = lt::Tensor{lt::Tensor::fromBuffer(c_data, lt::Shape(c_sh))};
    }

    auto st = std::chrono::steady_clock::now();
    auto res = lt::conv2d(a, b, c);
    auto ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(res, out_data, out_data_n, out_shape, out_shape_n);

    return dsec.count();
}

double max_pool2d(const float* a_data, const unsigned int a_data_n,
                  const int* a_shape, const unsigned int a_shape_n,
                  const int* ks, const unsigned int ks_n,
                  float** out_data, unsigned int* out_data_n,
                  int** out_shape, unsigned int* out_shape_n) {
    lt::manage::setDefaultGate<lt::CPUTensor>();

    std::vector<lt::dim_t> a_sh{a_shape, a_shape + a_shape_n}, 
                            ks_dat{ks, ks + ks_n};

    lt::Tensor a(lt::Tensor::fromBuffer(a_data, lt::Shape(a_sh)));
    lt::Shape kernel_shape{ks_dat};

    auto st = std::chrono::steady_clock::now();
    auto res = lt::max_pool2d(a, kernel_shape);
    auto ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(res, out_data, out_data_n, out_shape, out_shape_n);

    return dsec.count();
}
}