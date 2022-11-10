cmake_minimum_required(VERSION ${CMAKE_MIN_VER} FATAL_ERROR)

set(lib_name "cuda_utils")

#GLOB source files
file(GLOB_RECURSE CUH_FILES *.cuh)
file(GLOB_RECURSE HPP_FILES *.hpp)
file(GLOB_RECURSE CPP_FILES *.cu)

#Executable
add_library(${lib_name} ${HPP_FILES} ${CUH_FILES} ${CPP_FILES})

target_include_directories(${lib_name}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_utils"
)
