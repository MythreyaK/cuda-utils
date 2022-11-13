
#ifndef CUDA_DEVICE_VECTOR_CUH
#define CUDA_DEVICE_VECTOR_CUH

#include "allocator.cuh"
#include "misc.cuh"

namespace cuda {

    // outer_alloc is where the vector's data lives
    // inner_alloc is where T* points to

    template<typename T,
             typename alloc        = cuda::managed_allocator<T>,
             memory_type heap_type = memory_type::managed>
    class vector
      : public new_delete<vector<T, alloc, heap_type>,
                          cuda::allocator<vector<T, alloc, heap_type>, heap_type>> {
      public:
        T*     mem {};
        size_t m_size {};
        size_t m_capacity {};

      public:
        using allocator_t = alloc;

        // // this is where this container itself lives (m_size ..)
        // static constexpr cuda::memory_type outer_mem = outer_alloc::memory_t;

        // // this where data that T* points to lives
        static constexpr cuda::memory_type inner_mem = allocator_t::memory_t;

        HOST_DEVICE size_t size() const {

            if constexpr ( cuda::is_device_code() ) {
                return m_size;
            }
            else {
                // host call
                return cuda::get_value(&m_size);
            }
        }

        HOST_DEVICE void reserve(size_t items) {
            allocator_t alloc {};
            mem = alloc.allocate(sizeof(T) * items);

            if constexpr ( cuda::is_device_code() ) {
                m_size     = 0;
                m_capacity = items;
            }
            else {
                cuda::set_value(&m_size, 0ull);
                cuda::set_value(&m_capacity, items);
            }
        }

        HOST_DEVICE T* data() const {
            return mem;
        }

        HOST_DEVICE void push_back(const T& item) {
            // we don't need r-value and other overloads, it's copy to GPU mem
            // anyway. With managed/unified memory, there are potential
            // optimizations, but we'll deal with that later
            if constexpr ( cuda::is_device_code() ) {
                // parallel/concurrent access by default, we need to do sync so
                // fail for now
                static_assert(false,
                              "push_back is not yet supported in device_code");
            }
            else {
                // CPU world, it's understood sync is user's problem, and we are
                // usually running on a single thread
                cuda::set_value(mem[m_size], item);
                cuda::set_value(m_size, m_size + 1);
            }
        }

        HOST_DEVICE size_t capacity() const {
            if constexpr ( cuda::is_device_code() ) {
                return m_capacity;
            }
            else {
                // host call
                return cuda::get_value(m_capacity);
            }
        }

        HOST_DEVICE T& operator[](size_t inx) {
            // if T* is only on device, using [] on host is not valid
            if constexpr ( cuda::is_host_code()
                           && inner_mem == memory_type::device_local )
            {
                static_assert(false,
                              "Cannot use operator[] in host code when using "
                              "device memory device_allocator<T>");
            }

            assert(inx < m_capacity);
            return mem[inx];
        }

        HOST_DEVICE const T& operator[](size_t inx) const {
            assert(inx < m_capacity);
            return mem[inx];
        }
    };

    // this vector's data lives ONLY on the device. Ideal for in-kernel allocations
    template<typename T>
    using device_vector =
      cuda::vector<T, cuda::device_allocator<T>, memory_type::device_local>;

    // this vector's data is on managed heap, and can be accessed on device too
    template<typename T>
    using host_vector =
      cuda::vector<T, cuda::device_allocator<T>, memory_type::managed>;

}  // namespace cuda

#endif
