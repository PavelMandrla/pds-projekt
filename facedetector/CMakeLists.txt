cmake_minimum_required(VERSION 3.16)
project(cuda_add LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(add STATIC
        add.h
        add.cu
        )
set_target_properties( add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cuda_add main.cpp)

target_link_libraries(cuda_add
        ${OpenCV_LIBS}
        add
        )