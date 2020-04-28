cmake_minimum_required(VERSION 3.16)
project(dataset_maker LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(add STATIC
        #add.h
        #add.cu
        HistogramExtractor.h
        HistogramExtractor.cpp
        HistogramExtractor.cu
        ProcessedImage.h
        definitions.h
        )
set_target_properties( add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(dataset_maker main.cpp)

target_link_libraries(dataset_maker
        ${OpenCV_LIBS}
        add
)