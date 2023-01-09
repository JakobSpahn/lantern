#pragma once

#include "lantern/tensor/TensorGate.h"
#include "lantern/tensor/Shape.h"

#include <memory>

namespace lt {

using data_t = float;

class CUDATensor : public TensorGate {
 private:
    data_t* arr_;
    Shape sh;

 public:
    CUDATensor() = default;
    ~CUDATensor();

    CUDATensor(
        const void* dat, 
        const Shape& s, 
        const lt::dtype dt);
    
    std::unique_ptr<TensorGate> clone() override;
    void assign(const Tensor& t) override;

    Tensor copy() override;
    Tensor shallowCopy() override;

    TensorBackend& backend() const override;
    const Shape& shape() const override; 

    Tensor index(const Shape& sh) const override;

    std::string toString() const override;

    void buff(void** out) const override;

    data_t* data();
    const data_t* data() const;
};

}  // namespace lt