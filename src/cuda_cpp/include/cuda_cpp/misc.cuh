#ifndef CUDA_MISC_UTILS_CUH
#define CUDA_MISC_UTILS_CUH

#include "defs.cuh"

#include <cassert>
#include <iostream>
// #pragma nv_diag_suppress 20096

namespace cuda {
    // clang-format off

    // This works because of code-splitting that nvcc does!
    HOSTDEVICE
    consteval bool is_device_code() {
    #if ( !defined(__CUDA_ARCH__) )
        return false;
    #elif ( defined(__CUDA_ARCH__) )
        return true;
    #else
        #pragma error("__CUDA_ARCH__ is undefined. Cannot build host/device code")
    #endif
    }

    HOSTDEVICE
    consteval bool is_host_code() {
        return !cuda::is_device_code();
    }
    // clang-format on

}  // namespace cuda

#define CUDA_CHECK(X)                                                          \
    do {                                                                       \
        const auto ret_code = X;                                               \
        if ( ret_code != cudaSuccess ) {                                       \
            printf("%s:%d: CUDA ERROR: %s\n",                                  \
                   __FILE__,                                                   \
                   __LINE__,                                                   \
                   cudaGetErrorString(ret_code));                              \
            assert(ret_code == cudaSuccess);                                   \
        }                                                                      \
    } while ( 0 )

namespace cuda {

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
    void set_value(T& ptr, const T& value) {
        using mtype = cudaMemoryType;

        cudaPointerAttributes attr;

        CUDA_CHECK(cudaPointerGetAttributes(&attr, &ptr));

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
                ptr = value;
            }
        }
        else {
            // we're being called from host code, so if the ptr is not managed,
            // cudaMemcpy to device
            if ( attr.type == mtype::cudaMemoryTypeDevice ) {
                // do a copy
                CUDA_CHECK(
                  cudaMemcpy(&ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
            }
            else {
                // else it's managed or on host already, safe to dereference
                ptr = value;
            }
        }
    }

}  // namespace cuda

#include "undefs.cuh"
#endif
