#pragma once

#include "lantern/tensor/Shape.h"
#include "lantern/tensor/Types.h"

#include <memory>
#include <ostream>


namespace lt {

class TensorGate;

class TensorBackend;

enum class TensorBackendType { CPU, CUDA };

/**
 * @brief Central tensor class.
 * 
 * All operations are based on the underlying TensorGate implementation,
 * which manages the backend.
 * 
 */
class Tensor {
    // The gate to the tensor implementation
    std::unique_ptr<TensorGate> gate_;

    /**
     * @brief Construct new tensor from raw data. 
     * 
     * For internal use only. Calls corresponding constructor of gate_ implementation.
     * 
     * @param dat void ptr to raw data
     * @param s shape
     * @param dt dtype
     */
    Tensor(const void* dat, const Shape& s, lt::dtype dt);

 public:
    // Name variable of the tensor
    std::string name;

    ~Tensor() = default;

    /**
     * @brief Construct empty tensor.
     * 
     * Construction handled by gate_ implementation.
     * 
     */
    Tensor();
    
    /**
     * @brief Copy constructor.
     * 
     * Calls copy constructor defined by gate_ implementation.
     * 
     * @param t tensor
     */
    Tensor(const Tensor& t);

    /**
     * @brief Move constructor.
     * 
     * Calls move constructor defined by gate_ implementation.
     * 
     * @param t xvalue tensor reference
     */
    Tensor(Tensor&& t);

    /**
     * @brief Constructor for gate. 
     * 
     * Calls default constructor of gate.
     * 
     * @param gate 
     */
    // Tensor(const TensorGate& gate);
    Tensor(std::unique_ptr<TensorGate> new_gate);

    /**
     * @brief Construct empty tensor with shape and of data type.
     *  
     * @param s shape
     * @param dt dtype
     */
    // explicit Tensor(const Shape& s, lt::dtype dt = lt::dtype::float32);

    /**
     * @brief Construct empty tensor with merely the data type.
     * 
     * @param dt dtype
     */
    // explicit Tensor(lt::dtype dt);

    /**
     * @brief Copy assignment operator.
     * 
     * TODO: check if ptr can be copied here.
     * 
     * @param t 
     * @return Tensor& 
     */
    Tensor& operator=(const Tensor& rsh) &;
    Tensor& operator=(const Tensor& rsh) &&;

    /**
     * @brief Create copy of Tensor from vector.
     * 
     * If no shape is specified, tensor is assumed to be flat.
     * 
     * @tparam T 
     * @param vec 
     * @param sh 
     * @return Tensor 
     */
    template <typename T>
    static Tensor fromVector(const std::vector<T>& vec, const Shape& sh) {
        return Tensor(vec.data(), sh, lt::dtypeMapping<T>::lt_type);
    }

    /**
     * @brief Create Tensor from buffer. Data is copied.
     * 
     * @tparam T 
     * @param buff 
     * @param sh 
     * @return Tensor 
     */
    template <typename T>
    static Tensor fromBuffer(const T* buff, const Shape& sh) {
        return Tensor(buff, sh, lt::dtypeMapping<T>::lt_type);
    }

    /**
     * @brief Returns a tensor filled with the scalar value 0, and the given shape.
     * 
     * @tparam T 
     * @param sh Shape
     * @return Tensor 
     */
    template <typename T>
    static Tensor zeros(const Shape& sh) {
        return fromVector(std::vector<T>(sh.elements(), 0), sh);
    }

    /**
     * @brief Get the backend to perform ops on.
     * 
     * @return TensorBackend& 
     */
    TensorBackend& backend() const;

    /**
     * @brief Get the Gate object.
     * 
     * @tparam T 
     * @return T& 
     */
    template <typename T>
    T& getGate() const {
        return *static_cast<T*>(gate_.get());
    }

    /**
     * @brief Returns a buffer to the on-device data of the tensor.
     * 
     * @tparam T 
     * @return T* pointer to the first element in the tensor
     */
    template <typename T>
    T* buff() const;

    /**
     * @brief Get shape of tensor.
     * 
     * @return Shape& 
     */
    const Shape& shape() const;
    
    /**
     * @brief Index into a view of the tensor. Returns a shallow copy of the tensor.
     * 
     * @param sh 
     * @return Tensor 
     */
    Tensor index(const Shape& sh) const; 

    /**
     * @brief Returns a human readable string representation of the tensor.
     * 
     * @return std::string 
     */
    std::string toString() const;

    /**
     * @brief Returns true if the tensor is empty.
     * 
     * @return true 
     * @return false 
     */
    bool isEmpty() const;
    
    /**
     * @brief Returns the number of elements denoted by the tensor.
     * 
     * @return dim_t 
     */
    dim_t elements() const;
    
    /**
     * @brief Returns the dimensionality of the tensor.
     * 
     * @return dim_t 
     */
    dim_t ndim() const;
};

/******************** COMPLIANCE ********************/
std::ostream& operator<<(std::ostream& os, const Tensor& t);


/******************** ML Operators ********************/
Tensor reshape(const Tensor& lhs, const Shape& sh);
Tensor transpose(const Tensor& lhs, const Shape& sh);
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor conv2d(const Tensor& lhs, const Tensor& k, const Tensor& b);
Tensor max_pool2d(const Tensor& lhs, const Shape& k_sh);
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor relu(const Tensor& lhs);
Tensor softmax(const Tensor& lhs);

}  // namespace lt
