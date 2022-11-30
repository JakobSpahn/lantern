cmake_minimum_required(VERSION 3.25.0)

project(lantern)

# Specify CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ops SHARED
                lantern/csrc/ops.cc
                lantern/tensor.cc
                lantern/ops.cc)
add_executable(main 
                    main.cc
                    lantern/tensor.cc
                    lantern/ops.cc)