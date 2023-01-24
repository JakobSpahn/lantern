#include "include/lantern.h"
#include "lantern/tensor/accel/cuda/CUDABackend.h"

#include <iostream>
#include <chrono>

#define WARMUPS 5

static void bnch_mm(const lt::Tensor& x, const lt::Tensor& y) {
    std::cout << "benchmarking matmul: \t" 
                << x.shape() << " x " << y.shape() 
                << " with WARMUPS: " << WARMUPS << std::endl;

    for(size_t i{0}; i < WARMUPS; ++i) {
        lt::matmul(x,y);
    }

    auto st = std::chrono::steady_clock::now();
    auto res = lt::matmul(x, y);
    auto ed = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> dsec = ed - st;

    std::cout << "\t\tdur: " << dsec.count() << " [ms]" << std::endl;
}

static void bnch_mm_tld(const lt::Tensor& x, const lt::Tensor& y) {
    lt::CUDABackend::getInstance().tile = true;
    std::cout << "TILED: ";
    bnch_mm(x, y);
    lt::CUDABackend::getInstance().tile = false;
}

static void bnch_conv2d(const lt::Tensor& x, const lt::Tensor& k, const lt::Tensor& b) {
    std::cout << "benchmarking conv2d: \t" 
                << x.shape() << " " << k.shape() 
                << " with WARMUPS: " << WARMUPS << std::endl;

    for(size_t i{0}; i < WARMUPS; ++i) {
        lt::conv2d(x,k,b);
    }

    auto st = std::chrono::steady_clock::now();
    auto res = lt::conv2d(x,k,b);
    auto ed = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> dsec = ed - st;

    std::cout << "\t\tdur: " << dsec.count() << " [ms]" << std::endl;
}

static void bnch_conv2dchw(const lt::Tensor& x, const lt::Tensor& k, const lt::Tensor& b) {
    lt::CUDABackend::getInstance().conv_use_chw = true;
    std::cout << "CHW: ";
    bnch_conv2d(x, k, b);
    lt::CUDABackend::getInstance().conv_use_chw = false;
}

int main() {
    lt::manage::setDefaultGate<lt::CUDATensor>();

    // matmul
    for(long long int sz{128}; sz <= 1024; sz*=2) {
        auto x(lt::Tensor::randn<float>({1, sz})), 
            y(lt::Tensor::randn<float>({sz, sz}));

        bnch_mm(x,y);
        bnch_mm_tld(x,y);
    }

    for(auto b_sz : {1}) {
        for(auto c_out : {1,3, 6}) {
            for(auto w : {28}) {
                auto x(lt::Tensor::randn<float>({b_sz, 6, w, w})), 
                    y(lt::Tensor::randn<float>({c_out,6,5,5})),
                    b(lt::Tensor::randn<float>({c_out}));
                bnch_conv2d(x,y,b);
                bnch_conv2dchw(x,y,b);
            }
        }
    }
}