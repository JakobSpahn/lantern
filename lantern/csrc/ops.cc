#include <chrono>

#include "../helpersp.h"
#include "../tensor.h"

void update_out_param(const Tensor& x, float** out_data, unsigned int* out_data_n, int** out_shape, unsigned int* out_shape_n) {
    float* new_data = new float[x.size()];
    std::copy(x.get_raw(), x.get_raw() + x.size(), new_data);

    int* new_shape = new int[x.shape.size()];
    std::copy(x.shape.begin(), x.shape.end(), new_shape);

    *out_data = new_data;
    *out_data_n = x.size();
    *out_shape = new_shape;
    *out_shape_n = x.shape.size();
}

void matmul(Tensor& a, const Tensor& b) { a.matmul(b); }
void conv2d(Tensor& a, const Tensor& b, const Tensor& c) { a.conv2d(b, c); }
void max_pool2d(Tensor& x, const shape_t& kernel_shape) { x.max_pool(kernel_shape); }

extern "C" {
double matmul(const float* a_data, const unsigned int a_data_n,
              const int* a_shape, const unsigned int a_shape_n,
              const float* b_data, const unsigned int b_data_n,
              const int* b_shape, const unsigned int b_shape_n,
              float** out_data, unsigned int* out_data_n,
              int** out_shape, unsigned int* out_shape_n) {
    Tensor a{a_data, a_data_n, shape_t{a_shape, a_shape + a_shape_n}};
    const Tensor b{b_data, b_data_n, shape_t{b_shape, b_shape + b_shape_n}};

    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
    matmul(a, b);
    std::chrono::steady_clock::time_point ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(a, out_data, out_data_n, out_shape, out_shape_n);

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
    Tensor a{a_data, a_data_n, shape_t{a_shape, a_shape + a_shape_n}};
    const Tensor b{b_data, b_data_n, shape_t{b_shape, b_shape + b_shape_n}};
    Tensor c;

    if(use_c) { 
        c = Tensor{c_data, c_data_n, shape_t{c_shape, c_shape + c_shape_n}};;
    }

    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
    conv2d(a, b, c);
    std::chrono::steady_clock::time_point ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(a, out_data, out_data_n, out_shape, out_shape_n);

    return dsec.count();
}

double max_pool2d(const float* a_data, const unsigned int a_data_n,
                  const int* a_shape, const unsigned int a_shape_n,
                  const int* ks, const unsigned int ks_n,
                  float** out_data, unsigned int* out_data_n,
                  int** out_shape, unsigned int* out_shape_n) {
    Tensor a{a_data, a_data_n, shape_t{a_shape, a_shape + a_shape_n}};
    shape_t kernel_shape{ks, ks + ks_n};

    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
    max_pool2d(a, kernel_shape);
    std::chrono::steady_clock::time_point ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    update_out_param(a, out_data, out_data_n, out_shape, out_shape_n);

    return dsec.count();
}
}