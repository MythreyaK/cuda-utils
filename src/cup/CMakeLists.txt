set(lib_name "cuda_cup")

#GLOB source files
file(GLOB_RECURSE CUH_FILES *.cuh)
file(GLOB_RECURSE HPP_FILES *.hpp)
file(GLOB_RECURSE CPP_FILES *.cu)

#Executable
add_library(${lib_name} ${HPP_FILES} ${CUH_FILES} ${CPP_FILES})
add_library("cup::cup" ALIAS "cuda_cup")

target_include_directories(${lib_name}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_cpp"
)

target_link_libraries(${lib_name} PUBLIC cup_developer_flags)
