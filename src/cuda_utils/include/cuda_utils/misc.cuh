#ifndef CUDA_MISC_UTILS_CUH
#define CUDA_MISC_UTILS_CUH

#define HOST        __host__
#define DEVICE      __device__
#define GLOBAL      __global__
#define HOST_DEVICE __host__ __device__

#include <cassert>
#include <iostream>

#define CUDA_CHECK(X)                                                          \
    do {                                                                       \
        const auto ret_code = X;                                               \
        if ( ret_code != cudaSuccess ) {                                       \
            printf("%s:%d: CUDA ERROR: %s\n",                                  \
                   __FILE__,                                                   \
                   __LINE__,                                                   \
                   cudaGetErrorString(ret_code));                              \
            if constexpr ( cuda::is_host_code() ) {                            \
                std::flush(std::cout);                                         \
                std::flush(std::cerr);                                         \
            }                                                                  \
            assert(ret_code == cudaSuccess);                                   \
        }                                                                      \
    } while ( 0 )

namespace cuda {

    // clang-format off

    // This works because of code-splitting that nvcc does!
    // default template paramater, just in case, to force the compiler to evaluate this
    // at compile time. consteval is C++20 feature, cuda does not yet support it
    template <size_t N = 0>
    __host__ __device__
    constexpr bool is_device_code(size_t i = N) {
#if ( !defined(__CUDA_ARCH__) )
        return false;
#elif ( defined(__CUDA_ARCH__) )
        return true;
#else
    #pragma error("__CUDA_ARCH__ is undefined. Cannot build host/device code")
#endif
    }

    template <size_t N = 0>
    __host__ __device__
    constexpr bool is_host_code(size_t i = N) {
        return !cuda::is_device_code();
    }
    // clang-format on

    __host__ __device__ void where_am_i() {
        if constexpr ( is_device_code() )
            printf("Where am I: Device\n");
        else
            printf("Where am I: Host\n");
    }

    // read out a T at ptr, and do the right thing to read it
    // if it's on device / host and is being accessed from
    // host / device
    template<typename T>
    T get_value(const T& ptr) {
        using mtype = cudaMemoryType;

        T                     val;
        cudaPointerAttributes attr;

        CUDA_CHECK(cudaPointerGetAttributes(&attr, &ptr));

        if ( attr.type == mtype::cudaMemoryTypeUnregistered ) {
            assert("Invalid memory type to get_value");
        }

        if constexpr ( is_device_code() ) {
            // we're being called from device code. If ptr is on device
            // (mallocManaged / malloc), nothing to do. If it's on host, illeal
            if ( attr.type == mtype::cudaMemoryTypeHost ) {
                assert("Invalid memory type to get_value");
            }
            else {
                return ptr;
            }
        }
        else {
            // we're being called from host code, so if the ptr is not managed,
            // copy to host and return
            if ( attr.type == mtype::cudaMemoryTypeDevice ) {
                // do a copy
                CUDA_CHECK(
                  cudaMemcpy(&val, &ptr, sizeof(T), cudaMemcpyDeviceToHost));
                return val;
            }
            else {
                // else it's managed or on host already, safe to dereference
                return ptr;
            }
        }
    }

    // TODO: atomicCAS required?
    template<typename T>
    void set_value(T* ptr, const T& value) {
        using mtype = cudaMemoryType;

        cudaPointerAttributes attr;

        CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

        if ( attr.type == mtype::cudaMemoryTypeUnregistered ) {
            assert("Invalid memory type to get_value");
        }

        if constexpr ( is_device_code() ) {
            // we're being called from device code. If ptr is on device
            // (mallocManaged / malloc), nothing to do. If it's on host, it's illegal
            if ( attr.type == mtype::cudaMemoryTypeHost ) {
                assert("Cannot read a host pointer on device\n");
            }
            else {
                *ptr = value;
            }
        }
        else {
            // we're being called from host code, so if the ptr is not managed,
            // cudaMemcpy to device
            if ( attr.type == mtype::cudaMemoryTypeDevice ) {
                // do a copy
                CUDA_CHECK(
                  cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
            }
            else {
                // else it's managed or on host already, safe to dereference
                *ptr = value;
            }
        }
    }

}  // namespace cuda

#endif
