#include "lantern/tensor/Util.h"

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

namespace lt {
#define LT_GENERATE_FOR_REAL(REAL)                                              \
template<>                                                                      \
std::vector<REAL> randVec(size_t n) {                                           \
    std::random_device rnd_device;                                              \
    std::mt19937 mersenne_engine {rnd_device()};                                \
    std::uniform_real_distribution<REAL> dist {-2.0, 2.0};                      \
                                                                                \
    auto gen = [&dist, &mersenne_engine](){                                     \
                   return dist(mersenne_engine);                                \
               };                                                               \
    std::vector<REAL> vec(n);                                                         \
    std::generate(begin(vec), end(vec), gen);                                   \
    return vec;                                                                  \
}
LT_GENERATE_FOR_REAL(float);
LT_GENERATE_FOR_REAL(double);
}  // namespace lt