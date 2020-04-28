cmake_minimum_required(VERSION 3.16)
project(cuda_add LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(add STATIC
        #add.h
        #add.cu
        HistogramExtractor.cpp
        HistogramExtractor.h
        ProcessedImage.cpp
        ProcessedImage.h
        definitions.h
        HistogramExtractor.cu KNNClassifier.cpp KNNClassifier.h KNNClassifier.cu)
set_target_properties( add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(facedetector main.cpp)

target_link_libraries(facedetector
        ${OpenCV_LIBS}
        add
        )