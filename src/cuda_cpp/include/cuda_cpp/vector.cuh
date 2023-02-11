
#ifndef CUDA_DEVICE_VECTOR_CUH
#define CUDA_DEVICE_VECTOR_CUH

#include "allocator.cuh"
#include "defs.cuh"
#include "iterator.cuh"
#include "misc.cuh"
#include "utility.cuh"

#include <iostream>
#include <iterator>
#include <vector>

namespace cuda {

    template<typename T,
             typename _alloc        = managed_allocator<T>,
             memory_type _heap_type = memory_type::managed>
    class vector
      : public new_delete<
          vector<T, _alloc, _heap_type>,
          cuda::allocator<vector<T, _alloc, _heap_type>, _heap_type>> {

        // TODO: Support the case where the data-members are only GPU side
      public:
        T*     mem { nullptr };
        size_t m_size {};
        size_t m_capacity {};

      public:
        using value_type        = T;
        using const_value_type  = const T;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::contiguous_iterator_tag;
        using pointer           = value_type*;
        using reference         = value_type&;
        using const_pointer     = const value_type*;
        using const_reference   = const value_type&;

        using allocator_t            = _alloc;
        using iterator               = contiguous_iterator<value_type>;
        using reverse_iterator       = std::reverse_iterator<iterator>;
        using const_iterator         = contiguous_iterator<const_value_type>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        // this is where this container itself lives (m_size ..)
        static constexpr cuda::memory_type outer_mem = _heap_type;

        // this where data that T* points to lives
        static constexpr cuda::memory_type inner_mem = allocator_t::memory_t;

        // ctors
        vector() = default;

        // copy

        HOSTDEVICE vector(const vector& rhs)
          : m_size { rhs.m_size }
          , m_capacity { rhs.m_capacity } {

            allocator_t alloc {};
            mem = alloc.allocate(rhs.m_size);
            CUDA_CHECK(cudaMemcpy(mem,
                                  rhs.mem,
                                  rhs.m_size * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
        }

        HOSTDEVICE vector& operator=(const vector& rhs) {
            vector copy { rhs };
            copy.swap(*this);
            return *this;
        }

        // HOST vector(const vector& rhs) = delete;

        // HOST vector& operator=(const vector& rhs) = delete;

        // move

        HOSTDEVICE vector(vector&& rhs) noexcept
          : mem { rhs.mem }
          , m_size { rhs.m_size }
          , m_capacity { rhs.m_capacity } {
            rhs.mem        = nullptr;
            rhs.m_size     = 0;
            rhs.m_capacity = 0;
        }

        HOSTDEVICE ~vector() noexcept {
            allocator_t alloc {};
            alloc.deallocate(mem);
        }

        HOSTDEVICE vector& operator=(vector&& other) noexcept {
            vector copy { std::move(other) };
            copy.swap(*this);
            return *this;
        }

        // conversion from std::vector
        explicit vector(const std::vector<T>& vec) {
            reserve(vec.size());
            CUDA_CHECK(cudaMemcpy(mem,
                                  vec.data(),
                                  vec.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
            // set size to capacity, indicating buffer is full
            cuda::set_value(m_size, m_capacity);
        }

        // assignment from std::vector
        void operator=(const std::vector<T>& vec) {
            // check to make sure there is no data already
            if ( mem != nullptr ) {
                std::cerr << "Warning: Assigning to cuda::vector that already "
                             "has data\n";
            }

            reserve(vec.size());
            CUDA_CHECK(cudaMemcpy(mem,
                                  vec.data(),
                                  vec.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
            // set size to capacity, indicating buffer is full
            m_size = m_capacity;
        }

        HOSTDEVICE size_t size() const {

            if constexpr ( cuda::is_device_code() ) {
                return m_size;
            }
            else {
                // host call
                return cuda::get_value(m_size);
            }
        }

        HOSTDEVICE void resize(size_t items) {
            reserve(items);
            m_size = items;
            // TODO: Fix construction and stuff
        }

        HOSTDEVICE void reserve(size_t items) {
            allocator_t alloc {};
            mem = alloc.allocate(sizeof(T) * items);

            if constexpr ( cuda::is_device_code() ) {
                m_size     = 0;
                m_capacity = items;
            }
            else {
                cuda::set_value(m_size, 0ull);
                cuda::set_value(m_capacity, items);
            }
        }

        HOSTDEVICE T* data() const {
            return mem;
        }

        HOSTDEVICE void push_back(const T& item) {
            // we don't need r-value and other overloads, it's copy to GPU mem
            // anyway. With managed/unified memory, there are potential
            // optimizations, but we'll deal with that later
            if constexpr ( cuda::is_device_code() ) {
                // parallel/concurrent access by default, we need to do sync so
                // fail for now
                // static_assert(!sizeof(T*),
                //               "push_back is not yet supported in device_code");
            }
            else {
                // CPU world, it's understood sync is user's problem, and we are
                // usually running on a single thread
                cuda::set_value(mem[m_size], item);
                cuda::set_value(m_size, m_size + 1);
            }
        }

        HOSTDEVICE size_t capacity() const {
            if constexpr ( cuda::is_device_code() ) {
                return m_capacity;
            }
            else {
                // host call
                return cuda::get_value(m_capacity);
            }
        }

        HOSTDEVICE T& operator[](size_t inx) {
            // if T* is only on device, using [] on host is not valid
            if constexpr ( cuda::is_host_code()
                           && inner_mem == memory_type::device_local )
            {
                static_assert(!sizeof(T*),
                              "Cannot use operator[] in host code when using "
                              "device memory device_allocator<T>");
            }

            if ( inx >= m_capacity ) printf("inx: %llu\n", inx);
            assert(inx < m_capacity);
            return mem[inx];
        }

        HOSTDEVICE const T& operator[](size_t inx) const {
            // if T* is only on device, using [] on host is not valid
            if constexpr ( cuda::is_host_code()
                           && inner_mem == memory_type::device_local )
            {
                static_assert(!sizeof(T*),
                              "Cannot use operator[] in host code when using "
                              "device memory device_allocator<T>");
            }

            if ( inx >= m_capacity ) printf("inx: %llu\n", inx);
            assert(inx < m_capacity);
            return mem[inx];
        }

        HOSTDEVICE constexpr size_t bytes() noexcept {
            return sizeof(T) * m_size;
        }

        HOSTDEVICE friend vector& swap(vector& a, vector& b) noexcept {
            a.swap(b);
        }

        HOSTDEVICE vector& swap(vector& rhs) {
            using cuda::swap;
            swap(mem, rhs.mem);
            swap(m_size, rhs.m_size);
            swap(m_capacity, rhs.m_capacity);
            return *this;
        }

        // in the vector
        DEVICE void warp_sync(uint32_t mask, int src_thread) {
            m_size     = __shfl_sync(mask, m_size, src_thread);
            m_capacity = __shfl_sync(mask, m_capacity, src_thread);
            mem        = (T*)__shfl_sync(mask, (uint64_t)mem, src_thread);
        }

        // Iterator stuff
        constexpr iterator begin() {
            return { &mem[0] };
        }

        constexpr const_iterator cbegin() const {
            return { &mem[0] };
        }

        constexpr iterator end() {
            return { &mem[m_size] };
        }

        constexpr const_iterator cend() const {
            return { &mem[m_size] };
        }

        constexpr iterator front() {
            return begin();
        }

        constexpr iterator back() {
            return end() - difference_type { 1 };
        }

        constexpr const_iterator cfront() const {
            return begin();
        }

        constexpr const_iterator cback() const {
            return end() - difference_type { 1 };
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

#include "undefs.cuh"
#endif
