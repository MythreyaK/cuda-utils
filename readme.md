# CUDA-CPP

Tools under development to improve cuda experience in C++

Tested only on Windows 

```cpp
#include "cuda_utils/cuda_utils.cuh"

#include <iostream>
#include <vector>

template<typename T1, typename T2>
__global__ void kernel(cuda::vector<T1, T2>& ints) {

    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if ( thread_id == 0 ) {
        cuda::where_am_i();  // prints "Where am I: Device"

        // per-thread dynamic allocation on device
        cuda::device_vector<int> vec {};
        vec.reserve(blockDim.x);
        vec[2]  = 430;
        ints[2] = vec[2];

        // device side operators
        for ( int i = 0; i < ints.capacity(); ++i ) {
            printf("vector from host [%d]: %d\n", i, ints[i]);
        }

        // device side operators
        for ( int i = 0; i < vec.capacity(); ++i ) {
            printf("   device_vector [%d]: %d\n", i, vec[i]);
        }
    }
}

int main() {
    cuda::where_am_i();  // prints "Where am I: Host"

    // just works! ints are allocated using cudaMallocManaged
    auto std_vec = std::vector<int, cuda::managed_allocator<int>>();
    std_vec.reserve(8);

    auto  vec { std::make_unique<cuda::vector<int>>() };
    auto& vec_ref { *vec };

    vec_ref.reserve(4);

    vec_ref[1] = -1;

    auto t = cuda::time_it([&vec_ref]() {
        // pass the whole datastructure by reference, just works!
        kernel<<<1, 4>>>(vec_ref);

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    printf("Kernel took: %7.3fms\n", t);

    for ( int i = 0; i < vec_ref.capacity(); ++i ) {
        printf("vector [%d]: %d\n", i, vec_ref[i]);
    }
}

/* 
Output
Where am I: Host
Where am I: Device
vector from host [0]: 0
vector from host [1]: -1
vector from host [2]: 430
vector from host [3]: 0
   device_vector [0]: 0
   device_vector [1]: 0
   device_vector [2]: 430
   device_vector [3]: 0
Kernel took:   3.141ms
vector [0]: 0
vector [1]: -1
vector [2]: 430
vector [3]: 0
*/

```
