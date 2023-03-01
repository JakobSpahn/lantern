#pragma once

#include "lantern/tensor/TensorGate.h"
#include "lantern/tensor/Shape.h"

#include <memory>

namespace lt {

using data_t = float;

class CUDATensor : public TensorGate {
 private:
    void* arr_;
    Shape sh;
	dtype dt;

 public:
    CUDATensor() = default;
    ~CUDATensor();

    CUDATensor(
        const void* dat, 
        const Shape& s, 
        const lt::dtype dt);

    // shallow copy of dat
    CUDATensor(
        void* dat,
        const Shape& s);
    
    std::unique_ptr<TensorGate> clone() override;
    void assign(const Tensor& t) override;

    Tensor copy() override;
    Tensor shallowCopy() override;

    TensorBackend& backend() const override;
    const Shape& shape() const override; 

    Tensor index(const Shape& sh) const override;

    std::string toString() const override;

    void buff(void** out) const override;

	template <class T>
    T* data() {
		return static_cast<T*>(arr_);
	}

	template<class T>
    const T* data() const {
		return static_cast<T*>(arr_);
	}
};

}  // namespace lt