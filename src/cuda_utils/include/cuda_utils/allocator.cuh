#ifndef CUDA_MEMORY_ALLOCATOR_CUH
#define CUDA_MEMORY_ALLOCATOR_CUH

#include "misc.cuh"
#include "new_delete.cuh"

namespace cuda {

    template<typename T, memory_type MT>
    class allocator {
      public:
        static constexpr memory_type memory_t { MT };

        // type traits
        using value_type = T;

        template<typename TX>
        struct rebind {
            using other = allocator<TX, MT>;
        };

        constexpr allocator() = default;

        template<class TX, memory_type MT>
        constexpr allocator(const allocator<TX, MT>&) noexcept {}

        [[nodiscard]] HOST_DEVICE T* allocate(size_t byte_count) {
            T* ret = nullptr;

            if constexpr ( MT == memory_type::managed ) {

                if constexpr ( cuda::is_host_code() ) {
                    CUDA_CHECK(cudaMallocManaged(&ret, byte_count));
                }
                else if constexpr ( cuda::is_device_code() ) {
                    // unfortunately this can't be a compile time error (yet?
                    // hopefully!) due to how code-gen works :(
                    assert("Cannot call mallocManaged in device code");
                    // static_assert(!sizeof(T*), "Cannot call mallocManaged in
                    // device code ");
                }
            }
            else if constexpr ( MT == memory_type::device_local ) {
                // host or device, this is legal
                CUDA_CHECK(cudaMalloc(&ret, byte_count));
            }

            return ret;
        }

        HOST_DEVICE
        void deallocate(T* ptr) noexcept {
            CUDA_CHECK(cudaFree(ptr));
        }

        HOST_DEVICE
        void deallocate(T* ptr, size_t n) noexcept {
            CUDA_CHECK(cudaFree(ptr));
        }
    };

    template<class T, class U, cuda::memory_type MT>
    HOST_DEVICE bool operator==(const allocator<T, MT>&,
                                const allocator<U, MT>&) {
        return true;
    }

    template<class T, class U, cuda::memory_type MT>
    HOST_DEVICE bool operator!=(const allocator<T, MT>&,
                                const allocator<U, MT>&) {
        return false;
    }

    template<typename T>
    using managed_allocator = allocator<T, memory_type::managed>;

    template<typename T>
    using device_allocator = allocator<T, memory_type::device_local>;

}  // namespace cuda

#endif
