cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project("cup" LANGUAGES CXX C CUDA)

set(CMAKE_C_STANDARD    17)
set(CMAKE_CXX_STANDARD  23)
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_CUDA_FLAGS "-std=c++20")
endif()

#default build type is Release
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# # Auto-detect GPU architecture
set(CUDA_ARCHITECTURES native)


option(CUP_BUILD_EXAMPES "Build example" OFF)


# # CUDA and C++ compiler flags
# set(cxx_flags
#     $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp> #<TODO> Add MSVC-specific compiler flags here
#     $<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3>            #<TODO> Add GCC compiler flags here
#     $<$<CXX_COMPILER_ID:Clang>:-Wall -m64 -fopenmp -O3>          #<TODO> Add Clang compiler flags here
# )

set(cuda_flags
    # $<$<CXX_COMPILER_ID:GNU>:-Xcompiler -Wall -fopenmp -O3>
    # $<$<CXX_COMPILER_ID:Clang>:-Xcompiler -Wall -fopenmp -O3>
    # $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler>
    # -Xcudafe=--display_error_number
    # -lineinfo
    # -use_fast_math
    # $<$<CXX_COMPILER_ID:GNU>:-O3>
    --extended-lambda
    --expt-relaxed-constexpr
    # -Xptxas -warn-spills -res-usage
    # --ptxas-options=-v
    -G -g
)

add_library(cup_developer_flags INTERFACE)
target_compile_options(cup_developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

# target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory("src")
