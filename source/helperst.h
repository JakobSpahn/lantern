#pragma once


#include <numeric>
#include <vector>

using shape_t = std::vector<size_t>;

namespace t {

inline unsigned int mul_shape_elements(const shape_t& shape) {
    return std::accumulate(std::begin(shape), std::end(shape), 1,
                           std::multiplies<unsigned int>());
}

inline unsigned int mul_shape_elements(const shape_t::iterator beg,
                                       const shape_t::iterator end) {
    return std::accumulate(beg, end, 1, std::multiplies<unsigned int>());
}

}  // namespace t
