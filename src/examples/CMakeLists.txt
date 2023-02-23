
#GLOB source files
file(GLOB_RECURSE CUH_FILES *.cuh)
file(GLOB_RECURSE HPP_FILES *.hpp)
file(GLOB_RECURSE CPP_FILES *.cu)

#Executable
add_executable(cup_example_main ${HPP_FILES} ${CUH_FILES} ${CPP_FILES})

target_link_libraries(cup_example_main PUBLIC "cup::cup")

set_target_properties(cup_example_main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         native
)

target_link_libraries(cup_example_main PUBLIC cup_developer_flags)
