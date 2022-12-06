cmake_minimum_required(VERSION 3.18.4)

project(lantern)
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Specify CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_source_files_properties(lantern/csrc/cudaKernels.cu PROPERTIES LANGUAGE CUDA)
add_library(ops SHARED
                lantern/csrc/ops.cc
                lantern/tensor.cc
                lantern/ops.cc
                lantern/csrc/cudaKernels.cu)

set_property(TARGET ops 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ops
             PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET ops
             PROPERTY CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION ON)
                
add_executable(main 
                    main.cc
                    lantern/tensor.cc
                    lantern/ops.cc)
