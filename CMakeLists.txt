cmake_minimum_required(VERSION 3.25.0)

project(lantern)

# Specify CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(lantern_INCLUDE_DIRS
    #${CMAKE_SOURCE_DIR}/lantern/tensor/
    #${CMAKE_SOURCE_DIR}/lantern/csrc/)

add_library(ops SHARED
                lantern/csrc/ops.cc
                lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/accel/rawcpu/CPUBackend.cc
                lantern/tensor/accel/rawcpu/CPUTensor.cc)

add_executable(main 
                main.cc
                lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/accel/rawcpu/CPUBackend.cc
                lantern/tensor/accel/rawcpu/CPUTensor.cc)

#find_package(OpenMP REQUIRED)
#target_link_libraries(ops PRIVATE OpenMP::OpenMP_CXX)

include_directories(${CMAKE_SOURCE_DIR})