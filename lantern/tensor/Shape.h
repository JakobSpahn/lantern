#pragma once

#include <vector>
#include <ostream>
#include <string>

namespace lt {

// type alias for dimension
using dim_t = long long int;

class Shape {
    // Stores the dimensions of a tensor. 
    // Default initialized to empty vector {}.
    std::vector<dim_t> dims_;

    /**
     * @brief Bound checking for dim index.
     * 
     * @param dim 
     */
    void checkDimsOrThrow(const size_t dim) const;

 public:
    Shape() = default;
    ~Shape() = default;

    /**
     * @brief Construct a new Shape object via a vector.
     * 
     * @param d vector containing the dimensions
     */
    explicit Shape(std::vector<dim_t> d);

    /**
     * @brief Construct a new Shape object via an intializer list. 
     *
     * Allows for implicit construction, e.g. {1,2,3}.
     * 
     * @param d initializer list containing the dimensions
     */
    /* implicit */ Shape(std::initializer_list<dim_t>d);

    /**
     * @brief Number of elements described by the shape.
     * 
     * @return dim 
     */
    dim_t elements() const;

    /**
     * @brief Number of dimensions in the shape.
     * 
     * @return int 
     */
    size_t ndim() const;

    /**
     * @brief Returns the dimension of the shape at the given index.
     * 
     * @param idx index
     * @return dim 
     */
    dim_t dim(const size_t idx) const;

    /**
     * @brief Returns a reference to the dimension in the shape.
     * 
     * @param idx index
     * @return const dim_t& 
     */
    const dim_t& operator[](const size_t idx) const;
    /** See @ref const dim_t& operator[](const size_t idx) const "for more information"*/
    dim_t& operator[](const size_t idx);

    /**
     * @brief Compare with other shape
     * 
     * @param rhs 
     * @return true
     * @return false 
     */
    bool operator==(const Shape& rhs) const;
    /** See @ref bool operator==(const Shape& rhs) const "for more information" */
    bool operator!=(const Shape& rhs) const;
    /** See @ref bool operator==(const Shape& rhs) const "for more information" */
    bool operator==(const std::initializer_list<dim_t>& rhs) const;
    /** See @ref bool operator==(const Shape& rhs) const "for more information" */
    bool operator!=(const std::initializer_list<dim_t>& rhs) const;

    /**
     * @brief Get a refenrece to the underlying vector containing the dimensions.
     * 
     * @return reference to the underlying vector 
     */
    const std::vector<dim_t>& get() const;
    /** See @ref const std::vector<dim>& get() const "for more information"*/
    std::vector<dim_t>& get();

    /**
     * @brief Stringify the shape.
     * 
     * @return std::string copy of shape 
     */
    std::string toString() const;
};

/**
 * @brief Ostream interface for the shape.
 * 
 * Streams the stringified shape.
 * 
 * @param os reference of ostream to be streamed to
 * @param sh shape to be streamed into ostream
 * @return std::ostream& for operator chaining 
 */
std::ostream& operator<<(std::ostream& os, const Shape& sh);

} // namespace lt
