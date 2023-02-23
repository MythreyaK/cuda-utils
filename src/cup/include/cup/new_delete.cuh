#pragma once

#include "allocator.cuh"
#include "misc.cuh"

#include "defs.cuh"

namespace cup {

    // other overloads as needed
    // https://en.cppreference.com/w/cpp/memory/new/operator_new

    // new-delete for objects themselves, to be on cuda-managed heap
    // instead of CPU heap

    template<typename T, typename allocator>
    struct new_delete {
      public:
        [[nodiscard]] HOSTDEVICE static void*
        operator new(size_t byte_count) noexcept {
            allocator alloc {};
            // static_cast<T>(*this).new_de
            return alloc.allocate(byte_count);
        }

        HOSTDEVICE static void operator delete(void* ptr) noexcept {
            allocator alloc {};
            alloc.deallocate((T*)ptr);
        }

        [[nodiscard]] HOSTDEVICE static void*
        operator new[](size_t byte_count) noexcept {
            allocator alloc {};
            return alloc.allocate(byte_count);
        }

        HOSTDEVICE static void operator delete[](void* ptr) noexcept {
            allocator alloc {};
            alloc.deallocate((T*)ptr);
        }

        // other overloads as needed
        // https://en.cppreference.com/w/cpp/memory/new/operator_new
    };
}  // namespace cup

#include "undefs.cuh"
