#pragma once

#include <cstddef>
#include <string>

namespace lt {

enum class dtype {
    float32,  // 32-bit float
    float64,  // 64-bit float
    int8,     //  8-bit int
    int16,    // 16-bit int
    int32,    // 32-bit int
    int64,    // 64-bit int
    uint8,    //  8-bit unsigned int
    uint16,   // 16-bit unsigned int
    uint32,   // 32-bit unsigned int
    uint64,   // 64-bit unsigned int
    bool8     //  8-bit boolean
};

/**
 * @brief Struct used to map native ctype to lt::dtype.
 * 
 * Struct is defined with specializations for each lt::dtype.
 * 
 * @tparam T 
 */
template<typename T>
struct dtypeMapping;

/**
 * @brief Macro to help define each dtypeMapping specialization.
 * 
 */
#define LT_TYPE_MAPPING(C_TYPE, LT_TYPE, STR_TYPE) \
template <>                                        \
struct dtypeMapping<C_TYPE> {                      \
    typedef C_TYPE c_type;                         \
    static const dtype lt_type = LT_TYPE;          \
    static const char* to_string() {               \
        return STR_TYPE;                           \
    }                                              \
};                                                 

// Define specializations for dtypeMapping
LT_TYPE_MAPPING(float, dtype::float32, "float");
LT_TYPE_MAPPING(double, dtype::float64, "double");
LT_TYPE_MAPPING(char, dtype::int8, "char")
LT_TYPE_MAPPING(short, dtype::int16, "short");
LT_TYPE_MAPPING(int, dtype::int32, "int");
LT_TYPE_MAPPING(long long, dtype::int64, "long long");
LT_TYPE_MAPPING(unsigned char, dtype::uint8, "unsigned char");
LT_TYPE_MAPPING(unsigned short, dtype::uint16, "unsigned short");
LT_TYPE_MAPPING(unsigned int, dtype::uint32, "unsigned int");
LT_TYPE_MAPPING(unsigned long, dtype::uint64, "unsigned long long");
LT_TYPE_MAPPING(bool, dtype::bool8, "bool");

/**
 * @brief Get the name of dtype.
 * 
 * @param t dtype
 * @return std::string 
 */
std::string getTypeName(const dtype t);

/**
 * @brief Get the dtype for given name.
 * 
 * @param n name
 * @return dtype 
 */
dtype nameToType(const std::string& n);

/**
 * @brief Get the size of dtype.
 * 
 * @param t type
 * @return size_t 
 */
size_t getTypeSize(const dtype t);

template <typename T>
dtype getType() {
	return dtypeMapping<T>::lt_type;
}

} // namespace lt
