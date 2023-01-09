#include "lantern/tensor/Factory.h"

namespace lt{
namespace manage {

TensorDefaultGateManager& TensorDefaultGateManager::getInstance() {
    static TensorDefaultGateManager manager;
    return manager;
}

}  // namespace manage
}  // namespace lt