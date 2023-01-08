#include "Shape.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lt {

Shape::Shape(std::vector<dim_t> d): dims_(std::move(d)) {}
Shape::Shape(std::initializer_list<dim_t> d): Shape(std::vector<dim_t>(d)) {}

void Shape::checkDimsOrThrow(const size_t dim) const {
    if(dim > ndim() - 1) {
        std::stringstream ss;
        ss << "Shape index " << std::to_string(dim)
           << " out of bounds for shape with " << std::to_string(ndim())
           << " dimensions.";
        throw std::invalid_argument(ss.str());
    }
}

const dim_t emptyShapeNumElements = 0;

dim_t Shape::elements() const {
    if(dims_.size() == 0) {
        return emptyShapeNumElements;
    }
    return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<dim_t>());
}

size_t Shape::ndim() const {
    return dims_.size();
}

dim_t Shape::dim(const size_t idx) const {
    checkDimsOrThrow(idx);
    return dims_[idx];
}

const dim_t& Shape::operator[](const size_t idx) const {
    checkDimsOrThrow(idx);
    return dims_[idx];
}

dim_t& Shape::operator[](const size_t idx) {
    checkDimsOrThrow(idx);
    return dims_[idx];
}

bool Shape::operator==(const Shape& rhs) const {
    return dims_ == rhs.dims_;
}

bool Shape::operator!=(const Shape& rhs) const {
    return !((*this) == rhs);
}

bool Shape::operator==(const std::initializer_list<dim_t>& rhs) const {
    return dims_.size() == rhs.size()
           && std::equal(dims_.begin(), dims_.end(),
                         rhs.begin());
}

bool Shape::operator!=(const std::initializer_list<dim_t>& rhs) const {
    return !((*this) == rhs);
}

const std::vector<dim_t>& Shape::get() const {
    return dims_;
}

std::vector<dim_t>& Shape::get() {
    return dims_;
}

std::string Shape::toString() const {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < ndim(); ++i) {
        ss << dim(i) << (i == ndim() - 1 ? "" : ", ");
    }
    ss << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Shape& sh) {
    os << sh.toString();
    return os;
}

} // namespace lt
