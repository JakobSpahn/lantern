#pragma once
#include <cassert>
#include <cstring>
#include <numeric>
#include <ostream>
#include <vector>

#include "helperst.h"

#define assertm(exp, msg) assert(((void)msg, exp))

class Tensor {
    float* dat = nullptr;
    bool empty = true;

   public:
    shape_t shape;

    // constructors and destructor
    Tensor(shape_t shape);
    Tensor(const float* inp, unsigned int n, const shape_t& shape);
    Tensor(const Tensor& rhs);
    Tensor(const std::vector<float>& inp, shape_t new_shape);
    ~Tensor() { delete[] dat; }

    // operators
    Tensor& operator=(const Tensor& rhs);
    Tensor& operator=(const std::vector<float>& inp);
    float& operator[](const shape_t&);
    float operator[](const shape_t&) const;

    // tools
    bool isEmpty() { return empty; }
    std::ostream& print_shape(std::ostream& os);

    // ops
    void reshape(const shape_t& new_shape);
    void permute(const shape_t& permutation);
    void matmul(const Tensor& w);
    void conv2d(const Tensor& w, const Tensor& b,
                const std::string& padding = "valid");
    void max_pool(shape_t kernel_shape);
    void add(const Tensor& b);
    void relu();
    void softmax();
};
