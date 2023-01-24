cmake_minimum_required(VERSION 3.18.4)

project(lantern)

# Specify CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_package(CUDA)

set(core_sources lantern/tensor/Factory.cc
                lantern/tensor/Shape.cc
                lantern/tensor/Tensor.cc
                lantern/tensor/Types.cc
                lantern/tensor/Util.cc)
set(accel_sources lantern/tensor/accel/RuntimeCheck.cc) # this will be changed later on
set(cpu_sources lantern/tensor/accel/rawcpu/CPUBackend.cc
                lantern/tensor/accel/rawcpu/CPUTensor.cc)
set(cuda_sources lantern/tensor/accel/cuda/CUDATensor.cu
                 lantern/tensor/accel/cuda/CUDABackend.cc
                 lantern/tensor/accel/cuda/blas.cu
                 lantern/tensor/accel/cuda/kernel.cu)

set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(lantern/tensor/accel/cuda/optim/CUDATilingTensor.cu PROPERTIES LANGUAGE CUDA)

add_library(ops SHARED
                lantern/csrc/ops.cc
                ${core_sources}
                ${accel_sources}
                ${cpu_sources})

add_executable(main 
                main.cc
                ${core_sources}
                ${accel_sources}
                ${cpu_sources})

add_executable(benchmark 
                benchmark.cc
                ${core_sources}
                ${accel_sources}
                ${cpu_sources})


if(CUDA_FOUND)
enable_language(CUDA)

add_executable(cudaMain
                cudaMain.cc
                ${core_sources}
                ${accel_sources}
                ${cuda_sources})
target_sources(ops PRIVATE ${cuda_sources})
target_sources(benchmark PRIVATE ${cuda_sources})

set_property(TARGET cudaMain 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cudaMain
             PROPERTY CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)
set_property(TARGET ops 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ops
             PROPERTY CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)
add_compile_definitions(CUDA_)
endif()


#find_package(OpenMP REQUIRED)
#target_link_libraries(ops PRIVATE OpenMP::OpenMP_CXX)

include_directories(${CMAKE_SOURCE_DIR})