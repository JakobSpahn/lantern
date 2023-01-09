#include "lantern/tensor/Tensor.h"
#include "lantern/tensor/Factory.h"
#include "lantern/tensor/TensorGate.h"
#include "lantern/tensor/TensorBackend.h"

#include <utility>
#include <memory>

namespace lt {

Tensor::Tensor(
    const void* dat, 
    const Shape& s, 
    lt::dtype dt) 
    : gate_(lt::manage::getDefaultGate(dat, s, dt)) {}

Tensor::Tensor() : gate_(lt::manage::getDefaultGate()) {}

Tensor::Tensor(Tensor&& t) : gate_(std::move(t.gate_)) {}

Tensor::Tensor(const Tensor& t) : gate_(t.gate_->clone()) {}

Tensor::Tensor(std::unique_ptr<TensorGate> new_gate) : gate_(std::move(new_gate)) {}

// lvalue assginment operator
// gate_ is overwritten
Tensor& Tensor::operator=(const Tensor& rhs) & {
    gate_ = rhs.gate_->clone();
    return *this;
}

// rvalue assignment operator
// data is copied
Tensor& Tensor::operator=(const Tensor& rhs) && {
    gate_->assign(rhs);
    return *this;
}

TensorBackend& Tensor::backend() const {
    return gate_->backend();
}

/******************** TEMPLATE SPECIFICATIONS ********************/
#define LT_SPECIFY_OPS(TYPE)                        \
template <>                                         \
TYPE* Tensor::buff() const {                        \
    if(isEmpty()) {                                 \
        return nullptr;                             \
    }                                               \
    TYPE* ret;                                      \
    void** tmp = reinterpret_cast<void**>(&ret);    \
    gate_->buff(tmp);                               \
    return ret;                                     \
}
LT_SPECIFY_OPS(float);

/******************** UTILITY ********************/
const Shape& Tensor::shape() const {
    return gate_->shape();
}

Tensor Tensor::index(const Shape& sh) const {
    return gate_->index(sh);
}

std::string Tensor::toString() const {
    return gate_->toString();
}

bool Tensor::isEmpty() const {
    return shape().elements() < 1;
}

dim_t Tensor::elements() const {
    return shape().elements();
}

dim_t Tensor::ndim() const {
    return shape().ndim();
}

/******************** COMPLIANCE ********************/
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.toString();
    return os;
}

/******************** ML Operators ********************/
Tensor reshape(const Tensor& lhs, const Shape& sh) {
    return lhs.backend().reshape(lhs, sh);
}

Tensor transpose(const Tensor& lhs, const Shape& sh) {
    return lhs.backend().reshape(lhs, sh);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().matmul(lhs, rhs);
}

Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b) {
    return lhs.backend().conv2d(lhs, k, b);
}

Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh) {
    return lhs.backend().max_pool2d(lhs, k_sh);
}
}  // namespace lt