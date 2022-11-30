#include <chrono>

#include "../helpersp.h"
#include "../tensor.h"

struct ndarray {
    const float* data;
    unsigned int n_data;
    const int* shape;
    unsigned int n_shape;
};

enum OpType { MATMUL = 0, CONV2D = 1 };

void matmul(Tensor& a, const Tensor& b, const Tensor& c);
void conv2d(Tensor& a, const Tensor& b, const Tensor& c);

double binary_op(const ndarray& a, const ndarray& b, const ndarray& c,
                 const bool use_c, ndarray& out, const int type) {
    // init tensors (data is copied)
    Tensor t_a{a.data, a.n_data, shape_t(a.shape, a.shape + a.n_shape)};
    Tensor t_b{b.data, b.n_data, shape_t(b.shape, b.shape + b.n_shape)};
    Tensor t_c;

    if (use_c) {
        t_c = Tensor{c.data, c.n_data, shape_t{c.shape, c.shape + c.n_shape}};
    }

    void (*fxn)(Tensor & a, const Tensor& b, const Tensor& c);
    switch (type) {
        case MATMUL:
            fxn = matmul;
            break;
        case CONV2D:
            fxn = conv2d;
            break;
        default:
            printf("operation not implemented");
            break;
    }

    // perform op and measure
    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
    fxn(t_a, t_b, t_c);
    std::chrono::steady_clock::time_point ed = std::chrono::steady_clock::now();

    std::chrono::duration<double> dsec = ed - st;

    // update out param data
    float* new_data =
        reinterpret_cast<float*>(malloc(t_a.size() * sizeof(float)));
    std::memcpy(new_data, t_a.get_raw(), t_a.size() * sizeof(float));
    out.data = new_data;
    out.n_data = t_a.size();

    // update out param shape
    int* new_shape =
        reinterpret_cast<int*>(malloc(t_a.shape.size() * sizeof(int)));
    std::copy(t_a.shape.begin(), t_a.shape.end(), new_shape);
    out.shape = new_shape;
    out.n_shape = t_a.shape.size();

    return dsec.count();
}

void matmul(Tensor& a, const Tensor& b, const Tensor& c) {
    (void)c;

    a.matmul(b);
}

void conv2d(Tensor& a, const Tensor& b, const Tensor& c) { a.conv2d(b, c); }

extern "C" {
double binary_op(const float* a_data, const unsigned int a_data_n,
                 const int* a_shape, const unsigned int a_shape_n,
                 const float* b_data, const unsigned int b_data_n,
                 const int* b_shape, const unsigned int b_shape_n,
                 const float* c_data, const unsigned int c_data_n,
                 const int* c_shape, const unsigned int c_shape_n,
                 const bool use_c, float** out_data, unsigned int* out_data_n,
                 int** out_shape, unsigned int* out_shape_n, const int type) {
    // prepare params
    const ndarray a{a_data, a_data_n, a_shape, a_shape_n};
    const ndarray b{b_data, b_data_n, b_shape, b_shape_n};
    const ndarray c{c_data, c_data_n, c_shape, c_shape_n};
    ndarray out{*out_data, *out_data_n, *out_shape, *out_shape_n};

    // perform the test
    double dsec = binary_op(a, b, c, use_c, out, type);

    // update out params
    *out_data = const_cast<float*>(out.data);
    *out_data_n = out.n_data;
    *out_shape = const_cast<int*>(out.shape);
    *out_shape_n = out.n_shape;

    return dsec;
}
}