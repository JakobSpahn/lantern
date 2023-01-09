#pragma once

#include "lantern/tensor/Tensor.h"
#include "lantern/tensor/TensorBackend.h"
#include "lantern/tensor/Shape.h"
#include "lantern/tensor/Types.h"

namespace lt {

class TensorGate {
 public:
    TensorGate() = default;
    virtual ~TensorGate() = default;

    TensorGate(
        const void* dat, 
        const Shape& s, 
        lt::dtype dt);
    
    virtual std::unique_ptr<TensorGate> clone() = 0;
    virtual void assign(const Tensor& t) = 0;

    virtual Tensor copy() = 0;
    virtual Tensor shallowCopy() = 0;

    virtual TensorBackend& backend() const = 0;
    virtual const Shape& shape() const = 0; 

    virtual Tensor index(const Shape& sh) const = 0;

    virtual std::string toString() const = 0;

    virtual void buff(void** out) const = 0;
};

}  // namespace lt