#ifndef EXMPL_CUH
#define EXMPL_CUH

#include "cuda_utils/cuda_utils.cuh"

#include <cassert>

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
    cuda::device_vector<int> d_vec {};

    printf("item [%d]: %d\n", tid, ints[tid]);

    // create a new device vector here, and call test_vector again on this
    if ( tid == 0 ) {
        printf("Allocating vector of size %d in kernel\n", blockDim.x);
        d_vec.reserve(blockDim.x);
        d_vec[5] = 430;
    }

    __syncthreads();

    // use the same vec across all threads
    d_vec.m_capacity = __shfl_sync(0xffffffff, d_vec.m_capacity, 0);
    // sizeof(int*) is 64 bits! NOT 32! So can't use cast to int
    d_vec.mem = (int*)__shfl_sync(0xffffffff, (uint64_t)d_vec.mem, 0);

    __syncthreads();

    // printf("Tid: %d - %p %llu %llu\n", tid, d_vec.mem, d_vec.size(),
    // d_vec.capacity());

    d_vec[tid] = tid * 2;

    if ( tid == 0 )
        print_vec_kernel<<<1, d_vec.capacity()>>>(d_vec.data(),
                                                  d_vec.capacity());
}

__global__ void print() {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid == 0 ) {
        cuda::where_am_i();
    }
}

#endif
