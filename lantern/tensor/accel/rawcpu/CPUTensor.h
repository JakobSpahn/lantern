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

/*
template<lt::dtype> struct ltToBase;

#define LT_TO_BASE(C_TYPE, LT_TYPE) \
template <>                         \
struct ltToBase<LT_TYPE> {          \
    typedef C_TYPE c_type;          \
    dtype lt_type = LT_TYPE;        \
}; 

LT_TO_BASE(float, dtype::float32);
LT_TO_BASE(double, dtype::float64);
LT_TO_BASE(char, dtype::int8)
LT_TO_BASE(short, dtype::int16);
LT_TO_BASE(int, dtype::int32);
LT_TO_BASE(long long, dtype::int64);
LT_TO_BASE(unsigned char, dtype::uint8);
LT_TO_BASE(unsigned short, dtype::uint16);
LT_TO_BASE(unsigned int, dtype::uint32);
LT_TO_BASE(unsigned long, dtype::uint64);
LT_TO_BASE(bool, dtype::bool8);

template<lt::dtype type>
typename ltToBase<type>::c_type ltToBaseType() { return ltToBase<type>::c_type; }
*/

} // namespace lt
