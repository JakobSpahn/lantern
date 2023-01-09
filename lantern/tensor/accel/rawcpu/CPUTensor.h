#pragma once

#include "lantern/tensor/TensorGate.h"

#include <memory>


namespace lt
{

using data_t = float;

class CPUTensor : public TensorGate {
 private:
    std::shared_ptr<data_t[]> arr_;
    // std::unique_ptr<Shape> sh;
    Shape sh;
    lt::dtype dt;

    /*
    CPUTensor(
        std::shared_ptr<float[]> arr, 
        std::unique_ptr<Shape> sh, 
        lt::dtype dt);
    */

 public:
    CPUTensor() = default;
    ~CPUTensor() = default;

    CPUTensor(
        const void* dat, 
        const Shape& s, 
        const lt::dtype dt);
    
    // shallow copy of dat
    CPUTensor(
        std::shared_ptr<data_t[]> dat,
        const Shape& s);
    
    std::unique_ptr<TensorGate> clone() override;
    void assign(const Tensor& t) override;

    Tensor copy() override;
    Tensor shallowCopy() override;



    TensorBackend& backend() const override;
    const Shape& shape() const override; 

    Tensor index(const Shape& sh) const override;

    std::string toString() const override;

    std::shared_ptr<data_t[]> data();
    const std::shared_ptr<data_t[]> data() const;

    void buff(void** out) const override;
};

} // namespace lt
