#pragma once

#include "Tensor.h"
#include "TensorGate.h"

#include <memory>

namespace lt {

namespace manage {

class TensorGateFactory_ {
 public:
    TensorGateFactory_() = default;
    virtual ~TensorGateFactory_() = default;

    virtual std::unique_ptr<TensorGate> create(
        const void* dat = nullptr, 
        const Shape& s = {0}, 
        lt::dtype dt = lt::dtype::float32) const = 0;
};

template <typename T>
class TensorGateFactoryImpl_ : public TensorGateFactory_{
 public:
    TensorGateFactoryImpl_() = default;
    ~TensorGateFactoryImpl_() = default;

    std::unique_ptr<TensorGate> create(const void* dat, const Shape& s, lt::dtype dt) const override {
        return std::make_unique<T>(dat, s, dt);
    }
};

class TensorDefaultGateManager {
    // Object used to create the default TensorGate implementation. 
    // TensorGate implementation type is set via the template of TensorGateFactoryImpl_.
    std::unique_ptr<TensorGateFactory_> creator_;

    TensorDefaultGateManager() = default;
    ~TensorDefaultGateManager() = default;

 public:
    static TensorDefaultGateManager& getInstance();

    const TensorGateFactory_& getCreator() const {
        return *creator_;
    }

    void setCreator(std::unique_ptr<TensorGateFactory_> c) {
        creator_ = std::move(c);
    }

};

template <typename... T>
std::unique_ptr<TensorGate> getDefaultGate(T&&... t) {
return TensorDefaultGateManager::getInstance().getCreator().create(std::forward<T>(t)...);
}

template <typename T>
void setDefaultGate() {
    TensorDefaultGateManager::getInstance().setCreator(
        std::make_unique<TensorGateFactoryImpl_<T>>());
}

}  // namespace manage
}  // namespace lt
