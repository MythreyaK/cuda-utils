# Based on https://github.com/Ahdhn/CUDATemplate

#CMake minimum requirement

set(CMAKE_MIN_VER 3.20)
cmake_minimum_required(VERSION ${CMAKE_MIN_VER} FATAL_ERROR)

#Project name
set(project_name "CUDA Utils")

project(${project_name} LANGUAGES CXX C CUDA)

set(CMAKE_C_STANDARD    17)
set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CUDA_STANDARD 17)

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Auto-detect GPU architecture
include("cmake/AutoDetectCudaArch.cmake")

# CUDA and C++ compiler flags
set(cxx_flags
    $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp> #<TODO> Add MSVC-specific compiler flags here
    $<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3>            #<TODO> Add GCC compiler flags here
    $<$<CXX_COMPILER_ID:Clang>:-Wall -m64 -fopenmp -O3>          #<TODO> Add Clang compiler flags here
)

set(cuda_flags
    $<$<CXX_COMPILER_ID:GNU>:-Xcompiler -Wall -fopenmp -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Xcompiler -Wall -fopenmp -O3>
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler>
    -Xcudafe=--display_error_number
    -lineinfo
    --extended-lambda
    -use_fast_math
    $<$<CXX_COMPILER_ID:GNU>:-O3>
    --expt-relaxed-constexpr
    -Xptxas -warn-spills -res-usage
    --ptxas-options=-v
    -G
)

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory("src")
