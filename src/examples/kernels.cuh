#ifndef EXMPL_CUH
#define EXMPL_CUH

#include "cuda_cpp/concepts.cuh"
#include "cuda_cpp/cuda_cpp.cuh"
#include "cuda_cpp/utility.cuh"

#include <cassert>

HOSTDEVICE void where_am_i() {
    if constexpr ( cuda::is_device_code() )
        printf("Where am I: Device\n");
    else
        printf("Where am I: Host\n");
}

__global__ void print_vec(int* ints, size_t size) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("ints_vec [%d]: %d\n", tid, ints[tid]);
}

__global__ void print_vec_kernel(int* ints, size_t size) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf(" In kernel DP -> ints_vec [%d]: %d\n", tid, ints[tid]);
}

template<typename T1, typename T2>
__global__ void test_vector(cuda::vector<T1, T2>& ints) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid == 5 ) ints[tid] = 50;
    // create a device_vector, this is thread local!
    // any modifying operations are on thread 0
    cuda::warp_wide<cuda::device_vector<int>> d_wvec { 0 };

    printf("item [%d]: %d\n", tid, d_wvec->size());

    // create a new device vector here, and call test_vector again on this
    // use the same vector across all threads, so we need to warp-sync it
    d_wvec.apply_sync([](auto& vec) {
        printf("Allocating vector of size %d in kernel\n", blockDim.x);
        vec.resize(blockDim.x);
        vec[5] = 430;
    });

    // can only get read-only naked refernces, so have to go through
    // apply to modify the item across warps
    auto& d_vec = d_wvec.get();
    printf("item mod [%d]: %d\n", tid, d_vec[tid]);

    printf("Tid: %d - %p %llu %llu\n",
           tid,
           d_vec.mem,
           d_vec.size(),
           d_vec.capacity());

    // if ( tid == 0 )
    //     print_vec_kernel<<<1, d_vec.capacity()>>>(d_vec.data(),
    //                                               d_vec.capacity());
}

__global__ void print() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid == 0 ) {
        where_am_i();
    }
}

#endif
