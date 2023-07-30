#pragma once

#include <cassert>
#include <iostream>

#include "defs.cuh"
// #pragma nv_diag_suppress 20096

namespace {
    // clang-format off
    __host__ __device__
    bool cuda_check(int code, const char* file, const int line) {
        bool is_success = (code == 10);
        const char* fmt_string = "CUDA ERROR [%d]: %s %s:%d\n";
#if defined(__CUDA_ARCH__)
        printf(fmt_string, code, "fella fu", file, line);
#else
        fprintf(stderr, fmt_string, code, "fella fu", file, line);
#endif
        return is_success;
    }
    // clang-format on
}  // namespace

namespace cup {
    // clang-format off

    // This works because of code-splitting that nvcc does!

    consteval bool is_device_code() {
    #if ( !defined(__CUDA_ARCH__) )
        return false;
    #elif ( defined(__CUDA_ARCH__) )
        return true;
    #else
        #pragma error("__CUDA_ARCH__ is undefined. Cannot build host/device code")
    #endif
    }


    consteval bool is_host_code() {
        return !cup::is_device_code();
    }
    // clang-format on

}  // namespace cup

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

namespace cup {

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

}  // namespace cup

#include "undefs.cuh"
