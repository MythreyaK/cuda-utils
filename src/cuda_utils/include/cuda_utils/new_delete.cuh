#ifndef CUDA_NEW_DELETE_CUH
#define CUDA_NEW_DELETE_CUH

#include "misc.cuh"

namespace cuda {

    enum class memory_type : uint8_t {
        managed,
        device_local,
    };

    // other overloads as needed
    // https://en.cppreference.com/w/cpp/memory/new/operator_new

    // new-delete for objects themselves, to be on cuda-managed heap
    // instead of CPU heap
    template<typename T, typename allocator>
    struct new_delete {
      public:
        [[nodiscard]] HOST_DEVICE static void*
        operator new(size_t byte_count) noexcept {
            allocator alloc {};
            return alloc.allocate(byte_count);
        }

        HOST_DEVICE static void operator delete(void* ptr) noexcept {
            allocator alloc {};
            alloc.deallocate((T*)ptr);
        }

        [[nodiscard]] HOST_DEVICE static void*
        operator new[](size_t byte_count) noexcept {
            allocator alloc {};
            return alloc.allocate(byte_count);
        }

        HOST_DEVICE static void operator delete[](void* ptr) noexcept {
            allocator alloc {};
            alloc.deallocate((T*)ptr);
        }

        // other overloads as needed
        // https://en.cppreference.com/w/cpp/memory/new/operator_new
    };
}  // namespace cuda

#endif
