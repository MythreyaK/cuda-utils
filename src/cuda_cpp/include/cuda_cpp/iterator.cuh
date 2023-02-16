
#ifndef CUDA_ITERATOR_CUH
#define CUDA_ITERATOR_CUH

#include <concepts>
#include <iterator>
#include <type_traits>

#include "defs.cuh"
// #pragma nv_diag_suppress 2361

namespace cuda {

    template<typename T>
    struct contiguous_iterator {

        using value_type        = std::decay_t<T>;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::contiguous_iterator_tag;
        using pointer           = value_type*;
        using reference         = value_type&;
        using const_pointer     = const value_type*;
        using const_reference   = const value_type&;

        // clang-format off
        contiguous_iterator() = default;
        HOSTDEVICE contiguous_iterator(pointer m) : m_ptr { m } {}
        // clang-format on

        contiguous_iterator(const contiguous_iterator&)            = default;
        contiguous_iterator& operator=(const contiguous_iterator&) = default;

        contiguous_iterator(contiguous_iterator&&)            = default;
        contiguous_iterator& operator=(contiguous_iterator&&) = default;

        constexpr reference operator*() {
            return *m_ptr;
        }

        constexpr reference operator*() const {
            return *m_ptr;
        }

        constexpr pointer operator->() {
            return m_ptr;
        }

        constexpr pointer operator->() const {
            return m_ptr;
        }

        constexpr reference operator[](size_t inx) {
            return m_ptr[inx];
        }

        constexpr reference operator[](size_t inx) const {
            return m_ptr[inx];
        }

        // ADD

        constexpr contiguous_iterator& operator++() {
            m_ptr++;
            return *this;
        }

        constexpr contiguous_iterator operator+(const difference_type& d) const {
            return m_ptr + d;
        }

        constexpr contiguous_iterator operator++(int) {
            contiguous_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr contiguous_iterator& operator+=(size_t skip) {
            m_ptr += skip;
            return *this;
        }

        // SUB

        constexpr contiguous_iterator& operator--() {
            m_ptr--;
            return *this;
        }

        constexpr difference_type operator-(const contiguous_iterator& d) const {
            return m_ptr - d.m_ptr;
        }

        constexpr contiguous_iterator operator-(const difference_type& d) const {
            return m_ptr - d;
        }

        constexpr contiguous_iterator operator--(int) {
            contiguous_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        constexpr contiguous_iterator& operator-=(size_t skip) {
            m_ptr -= skip;
            return *this;
        }

        // Others

        std::strong_ordering
        operator<=>(const contiguous_iterator& o) const = default;

        friend constexpr bool operator==(const contiguous_iterator& a,
                                         const contiguous_iterator& b) {
            return a.m_ptr == b.m_ptr;
        };

        friend constexpr bool operator!=(const contiguous_iterator& a,
                                         const contiguous_iterator& b) {
            return a.m_ptr != b.m_ptr;
        };

        friend constexpr contiguous_iterator
        operator+(const difference_type& a, const contiguous_iterator& b) {
            return a + b.m_ptr;
        };

        //   private:
        pointer m_ptr {};
    };

    template<typename T>
    struct grid_stride_iterator {
        T m_pos {};

        // using value_type        = std::decay_t<typename T::value_type>;
        // using const_value_type  = const T;
        // using difference_type   = std::ptrdiff_t;
        // using iterator_category = std::contiguous_iterator_tag;
        // using pointer           = value_type*;
        // using reference         = value_type&;
        // using const_pointer     = const value_type*;
        // using const_reference   = const value_type&;

        // using iterator = T::iterator;
        // // using reverse_iterator       = std::reverse_iterator<iterator>;
        // using const_iterator = const contiguous_iterator<const_value_type>;

        // this looks counter intuitive, but remember that each thread calls
        // begin, so each thread gets its own. Map it out to how one would
        // typically write out a grid-stride loop

        grid_stride_iterator() = default;

        DEVICE grid_stride_iterator(T pos)
          : m_pos { pos + threadIdx.x + blockIdx.x * gridDim.x } {
            // printf("Iterator constructed in thread: %2d\n",
            //        threadIdx.x + blockIdx.x * gridDim.x);
        }

        // DEVICE grid_stride_iterator(iterator begin, iterator end)
        //   : m_thisthread { threadIdx.x + blockIdx.x * gridDim.x }
        //   , m_stride { gridDim.x * blockDim.x }
        //   , m_begin { begin + m_thisthread }
        //   , m_end { end }
        //   , m_current { begin } {}

        // DEVICE grid_stride_iterator(const T& obj)
        //   : m_thisthread { threadIdx.x + blockIdx.x * gridDim.x }
        //   , m_stride { gridDim.x * blockDim.x }
        //   , m_begin { obj.cbegin() + m_thisthread }
        //   , m_end { obj.cend() }
        //   , m_current { obj.cbegin() } {}

        // DEVICE grid_stride_iterator(const_iterator begin, const_iterator
        // end)
        //   : m_thisthread { threadIdx.x + blockIdx.x * gridDim.x }
        //   , m_stride { gridDim.x * blockDim.x }
        //   , m_begin { begin + m_thisthread }
        //   , m_end { end }
        //   , m_current { begin } {}

        // WARNING: invalid narrowing conversion from "unsigned int" to
        // "int" but should be fine, right? We can't have more than 2^23-1
        // threads anyway (in m_start + m_thisthread)

        // these operator are "called" by the ranged-for-loop

        // constexpr iterator& operator++() {
        //     m_current += m_stride;
        //     if (m_current > m_end) m_current = m_end;
        //     return *this;
        // }

        DEVICE constexpr grid_stride_iterator& operator++() {
            m_pos += blockDim.x * gridDim.x;
            return *this;
        }

        DEVICE constexpr auto& operator*() {
            return *m_pos;
        }

        // constexpr grid_stride_iterator& operator++(int) {
        //     m_pos += blockDim.x * gridDim.x;
        //     return *this;
        // }

        constexpr std::strong_ordering
        operator<=>(const grid_stride_iterator& o) const = default;
    };

    // // TODO: stopgap to test things, flesh this out completely later
    // // currently supports only 1D mapping
    // // Also, ranges librrary support?

    template<typename T>
    // requires std::contiguous_iterator<T>
    struct grid_stride_adaptor {

        using value_type        = std::decay_t<typename T::value_type>;
        using const_value_type  = const T;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::contiguous_iterator_tag;

        // decltype teeheehe
        using _iterator       = decltype(std::declval<T>().begin());
        using _const_iterator = decltype(std::declval<T>().cbegin());
        using _return_t       = decltype(*std::declval<T>().begin());

        using iterator =
          grid_stride_iterator<std::conditional_t<std::is_const_v<_return_t>,
                                                  _const_iterator,
                                                  _iterator>>;

        T* wrapped_container {};

        DEVICE grid_stride_adaptor(T& container)
          : wrapped_container { &container } {}

        DEVICE constexpr iterator begin() {
            return { wrapped_container->begin() };
        }

        DEVICE constexpr iterator cbegin() {
            return { wrapped_container->cbegin() };
        }

        DEVICE constexpr iterator end() {
            auto st = blockDim.x * gridDim.x;
            return { wrapped_container->end()
                     + (wrapped_container->size() % st) };
        }

        DEVICE constexpr iterator cend() {
            auto st = blockDim.x * gridDim.x;
            return { wrapped_container->cend()
                     + (wrapped_container->size() % st) };
        }
    };

}  // namespace cuda

#include "undefs.cuh"
#endif
