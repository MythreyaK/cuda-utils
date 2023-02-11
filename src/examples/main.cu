#include "cuda_cpp/cuda_cpp.cuh"
#include "kernels.cuh"

#include <concepts>
#include <iostream>
#include <memory>
#include <vector>

int main() {
    {
        cuda::scoped_timer timer(
          [](float t) { printf("Time taken: %5.2fms\n", t); });

        // this prints host or device depending on where it's called from
        where_am_i();
        print<<<1, 16>>>();

        CUDA_CHECK(cudaDeviceSynchronize());

        // this is a managed vector by default, so can use [] on host
        // cuda runtime manages page-fault to keep things consistent
        auto  vec { std::make_unique<cuda::vector<int>>() };
        auto& vec_ref { *vec };

        vec_ref.resize(8);

        for ( int i = 0; i < vec_ref.size(); ++i )
            vec_ref[i] = i;

        vec_ref[3] = -1;

        auto t = cuda::time_it([&vec_ref]() {
            print_vec<<<1, 8>>>(vec_ref.mem, vec_ref.size());
            print_vec_iterator<<<2, 2>>>(vec_ref);

            CUDA_CHECK(cudaDeviceSynchronize());
        });

        printf("Kernel took: %7.3fms\n", t);

        // vec cleaned up here
    }

    printf(" ------------- \n");

    {
        auto vec {
            std::make_unique<cuda::vector<int, cuda::device_allocator<int>>>()
        };
        vec->reserve(1024);

        auto& vec_ref { *vec };

        // device local array cannot be accessed on host, so uncommenting
        // this should result in a compilation error!
        // vec_ref[3] = 10;

        test_vector<<<1, 8>>>(*vec);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf(" ------------- \n");

    {

        auto vec = std::vector<int, cuda::managed_allocator<int>>();
        vec.reserve(8);

        vec.push_back(0);
        vec.push_back(1);
        vec.push_back(2);
        vec.push_back(4);
        vec.push_back(8);
        vec.push_back(16);
        vec.push_back(32);
        vec.push_back(64);

        print_vec<<<1, vec.size()>>>(vec.data(), vec.size());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return 0;
}
