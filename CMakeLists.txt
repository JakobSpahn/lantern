cmake_minimum_required(VERSION 3.18.4)

project(lantern)

# Specify CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

#set(lantern_INCLUDE_DIRS
    #${CMAKE_SOURCE_DIR}/lantern/tensor/
    #${CMAKE_SOURCE_DIR}/lantern/csrc/)

set_source_files_properties(lantern/tensor/accel/cuda/CUDATensor.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(lantern/tensor/accel/cuda/CUDABackend.cu PROPERTIES LANGUAGE CUDA)

add_library(ops SHARED
                lantern/csrc/ops.cc
                lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/accel/RuntimeCheck.cc
                lantern/tensor/accel/rawcpu/CPUBackend.cc
                lantern/tensor/accel/rawcpu/CPUTensor.cc
                lantern/tensor/accel/cuda/CUDABackend.cu
                lantern/tensor/accel/cuda/CUDATensor.cu)

add_executable(main 
                main.cc
                lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/accel/RuntimeCheck.cc
                lantern/tensor/accel/rawcpu/CPUBackend.cc
                lantern/tensor/accel/rawcpu/CPUTensor.cc)

find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(cudaMain
                cudaMain.cc
                lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/accel/RuntimeCheck.cc 
                lantern/tensor/accel/cuda/CUDABackend.cu
                lantern/tensor/accel/cuda/CUDATensor.cu)

set_property(TARGET cudaMain 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cudaMain
             PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET cudaMain
             PROPERTY CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)

#find_package(OpenMP REQUIRED)
#target_link_libraries(ops PRIVATE OpenMP::OpenMP_CXX)

include_directories(${CMAKE_SOURCE_DIR})