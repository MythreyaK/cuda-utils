#pragma once

#include "concepts.cuh"

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "defs.cuh"
// #pragma nv_diag_suppress 2361

namespace {
    template<typename T>
    struct const_or_iterator {
        using value = T::iterator;
    };

    template<typename T>
    requires cup::has_const_iterator<T>
    struct const_or_iterator<T> {
        using value = typename T::const_iterator;
    };
}  // namespace

namespace cup {

    template<typename T>
    struct vec_contig_iterator {

        using value_type        = std::decay_t<T>;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::contiguous_iterator_tag;
        using pointer           = value_type* __restrict__;
        using reference         = value_type&;
        using const_pointer     = const value_type* __restrict__;
        using const_reference   = const value_type&;

        // clang-format off
        vec_contig_iterator() = default;
        HOSTDEVICE vec_contig_iterator(pointer m) : m_ptr { m } {}
        // clang-format on

        vec_contig_iterator(const vec_contig_iterator&)            = default;
        vec_contig_iterator& operator=(const vec_contig_iterator&) = default;

        vec_contig_iterator(vec_contig_iterator&&)            = default;
        vec_contig_iterator& operator=(vec_contig_iterator&&) = default;

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

        constexpr vec_contig_iterator& operator++() {
            m_ptr++;
            return *this;
        }

        constexpr vec_contig_iterator operator+(const difference_type& d) const {
            return m_ptr + d;
        }

        constexpr vec_contig_iterator operator++(int) {
            vec_contig_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr vec_contig_iterator& operator+=(size_t skip) {
            m_ptr += skip;
            return *this;
        }

        // SUB

        constexpr vec_contig_iterator& operator--() {
            m_ptr--;
            return *this;
        }

        constexpr difference_type operator-(const vec_contig_iterator& d) const {
            return m_ptr - d.m_ptr;
        }

        constexpr vec_contig_iterator operator-(const difference_type& d) const {
            return m_ptr - d;
        }

        constexpr vec_contig_iterator operator--(int) {
            vec_contig_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        constexpr vec_contig_iterator& operator-=(size_t skip) {
            m_ptr -= skip;
            return *this;
        }

        // Others

        std::strong_ordering
        operator<=>(const vec_contig_iterator& o) const = default;

        friend constexpr bool operator==(const vec_contig_iterator& a,
                                         const vec_contig_iterator& b) {
            return a.m_ptr == b.m_ptr;
        };

        friend constexpr bool operator!=(const vec_contig_iterator& a,
                                         const vec_contig_iterator& b) {
            return a.m_ptr != b.m_ptr;
        };

        friend constexpr vec_contig_iterator
        operator+(const difference_type& a, const vec_contig_iterator& b) {
            return a + b.m_ptr;
        };

      private:
        pointer m_ptr {};
    };

    template<typename T>
    struct grid_stride_iterator {

        using value_type        = std::decay_t<typename T::value_type>;
        using const_value_type  = const T;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::contiguous_iterator_tag;
        using pointer           = value_type* __restrict__;
        using reference         = value_type&;
        using const_pointer     = const value_type* __restrict__;
        using const_reference   = const value_type&;

        // using iterator = T::iterator;
        // // using reverse_iterator       = std::reverse_iterator<iterator>;
        // using const_iterator = const vec_contig_iterator<const_value_type>;

        // this looks counter intuitive, but remember that each thread calls
        // begin, so each thread gets its own. Map it out to how one would
        // typically write out a grid-stride loop

        grid_stride_iterator() = default;

        DEVICE grid_stride_iterator(T pos)
          : m_pos { &*pos + threadIdx.x + (blockIdx.x * blockDim.x) } {
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

        FORCEINLINE DEVICE constexpr grid_stride_iterator& operator++() {
            m_pos += blockDim.x * gridDim.x;
            return *this;
        }

        FORCEINLINE DEVICE constexpr auto operator&() {
            return m_pos;
        }

        FORCEINLINE DEVICE constexpr auto& operator*() {
            return *m_pos;
        }

        constexpr grid_stride_iterator& operator++(int) {
            m_pos += blockDim.x * gridDim.x;
            return *this;
        }

        FORCEINLINE constexpr std::strong_ordering
        operator<=>(const grid_stride_iterator& o) const = default;

      private:
        pointer m_pos {};
    };

    // TODO: might need more things, currently supports only 1D mapping
    // Also, ranges librrary support?

    template<typename T>
    requires std::contiguous_iterator<typename T::iterator>
    struct grid_stride_adaptor {

        using value_type       = std::decay_t<typename T::value_type>;
        using const_value_type = const T;
        using difference_type  = std::ptrdiff_t;
        // using iterator_category = std::contiguous_iterator_tag;
        using reference       = value_type&;
        using const_pointer   = const value_type* __restrict__;
        using const_reference = const value_type&;

        // decltype teeheehe
        // decltype(std::declval<T>().begin());
        using _iterator       = typename T::iterator;
        using _const_iterator = const_or_iterator<T>;
        using _return_t       = decltype(*std::declval<T>().begin());

        using underlying_iterator = std::
          conditional_t<std::is_const_v<_return_t>, _const_iterator, _iterator>;

        using iterator = grid_stride_iterator<underlying_iterator>;

        FORCEINLINE DEVICE grid_stride_adaptor(T& ctr)
          : pstart { ctr.begin() } {
            const size_t sz     = ctr.size();
            const size_t stride = blockDim.x * gridDim.x;
            const size_t remd   = sz % stride;
            const size_t tid = threadIdx.x + (blockIdx.x * blockDim.x);

            pend = pstart + sz - remd;
            if (remd != 0 && tid < remd)
                pend += stride;
        }

        DEVICE constexpr iterator begin() {
            return { pstart };
        }

        DEVICE constexpr iterator cbegin()
        requires has_const_iterator<T>
        {
            return { pstart };
        }

        DEVICE constexpr iterator end() {
            return { pend };
        }

        DEVICE constexpr iterator cend()
        requires has_const_iterator<T>
        {
            return { pend };
        }

      private:
        underlying_iterator pstart {};
        underlying_iterator pend {};
    };

}  // namespace cup

#include "undefs.cuh"
