cmake_minimum_required(VERSION ${CMAKE_MIN_VER} FATAL_ERROR)

set(exec_name "main")

#GLOB source files
file(GLOB_RECURSE CUH_FILES *.cuh)
file(GLOB_RECURSE HPP_FILES *.hpp)
file(GLOB_RECURSE CPP_FILES *.cu)

#Executable
add_executable(${exec_name} ${HPP_FILES} ${CUH_FILES} ${CPP_FILES})

target_link_libraries(${exec_name} PUBLIC "cuda_cpp")

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         native
)

target_link_libraries(${exec_name} PUBLIC developer_flags)
