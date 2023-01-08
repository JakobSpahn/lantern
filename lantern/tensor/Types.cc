/**
 * @file Types.h
 * @author Jakob Spahn (jakob@craalse.de)
 * @brief 
 * @date 2022-12-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "Types.h"

#include <string>
#include <unordered_map>
#include <stdexcept>

namespace lt {

const std::unordered_map<dtype, std::string> mapDtypeToString = {
    {dtype::float32, "float32"},
    {dtype::float64, "float64"},
    {dtype::int8, "int8"},
    {dtype::int16, "int16"},
    {dtype::int32, "int32"},
    {dtype::int64, "int64"},
    {dtype::uint8, "uint8"},
    {dtype::uint16, "uint16"},
    {dtype::uint32, "uint32"},
    {dtype::uint64, "uint64"},
    {dtype::bool8, "bool8"}
};

const std::unordered_map<std::string, dtype> mapStringToDtype = {
    {"float32", dtype::float32},
    {"float64", dtype::float64},
    {"int8", dtype::int8},
    {"int16", dtype::int16},
    {"int32", dtype::int32},
    {"int64", dtype::int64},
    {"uint8", dtype::uint8},
    {"uint16", dtype::uint16},
    {"uint32", dtype::uint32},
    {"uint64", dtype::uint64},
    {"bool8", dtype::bool8}
};

/**
 * @brief Get the name of dtype.
 * 
 * @param t dtype
 * @return std::string 
 */
std::string getTypeName(const dtype t) {
    return mapDtypeToString.at(t);
}

/**
 * @brief Get the dtype for given name.
 * 
 * @param n name
 * @return dtype 
 */
dtype nameToType(const std::string& n) {
    return mapStringToDtype.at(n);
}

/**
 * @brief Get the size of dtype.
 * 
 * @param t type
 * @return size_t 
 */
size_t getTypeSize(const dtype t) {
    switch (t) {
        case dtype::float32: return sizeof(float);
        case dtype::float64: return sizeof(double);
        case dtype::int8: return sizeof(char);
        case dtype::int16: return sizeof(short);
        case dtype::int32: return sizeof(int);
        case dtype::int64: return sizeof(long long);
        case dtype::uint8: return sizeof(unsigned char);
        case dtype::uint16: return sizeof(unsigned short);
        case dtype::uint32: return sizeof(unsigned int);
        case dtype::uint64: return sizeof(unsigned long long);
        case dtype::bool8: return sizeof(bool);
        default:
            std::invalid_argument("dtype not implemented");
            break;
    }
}

}  // namespace lt