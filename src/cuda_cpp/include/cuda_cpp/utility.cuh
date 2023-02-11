#ifndef CUDA_CPP_UTILITY_CUH
#define CUDA_CPP_UTILITY_CUH

#include "concepts.cuh"
#include "defs.cuh"

#include <concepts>
#include <type_traits>

namespace {

    // scalar types decay to a copy, others are moved
    // TODO: Is this the correct way or does this do what I expect?
    template<typename T>
    concept swappable = std::is_scalar_v<std::remove_reference_t<T>>
                     || std::movable<std::remove_reference_t<T>>;
}  // namespace

namespace cuda {

    template<typename T>
    requires warp_syncable<T>
    struct warp_wide {
        union {
            T wrapped {};
        };
        uint8_t owning_thread { 0 };

        template<typename... Args>
        DEVICE warp_wide(uint8_t owning_thread = 0, Args... args) {
            if ( threadIdx.x == owning_thread ) {
                wrapped = std::move(T(std::forward<Args>(args)...));
            }
        }

        DEVICE const T& get() {
            return wrapped;
        }

        HOSTDEVICE const T* operator->() {
            return &wrapped;
        }

        DEVICE void sync(uint32_t mask = __activemask()) {
            if constexpr ( cuda::native_warp_syncable<T> ) {
                // native type accepted by __shfl_sync
                wrapped = __shfl_sync(mask, wrapped, owning_thread);
            }
            else if constexpr ( sizeof(T) == sizeof(uint32_t) ) {
                // can cast it to uint32_t and sync it over
                wrapped =
                  (T)__shfl_sync(mask, (uint32_t)wrapped, owning_thread);
            }
            else if constexpr ( sizeof(T) == sizeof(uint64_t) ) {
                // can cast it to uint64_t and sync it over
                wrapped =
                  (T)__shfl_sync(mask, (uint64_t)wrapped, owning_thread);
            }
            else {
                // otherwise it needs to support a memberwise call
                wrapped.warp_sync(mask, owning_thread);
            }
        }

        template<typename Func>
        requires std::invocable<Func, T&>
        DEVICE warp_wide& apply(Func&& func) {
            if ( threadIdx.x == owning_thread ) {
                func(wrapped);
            }
            return *this;
        }

        template<typename Func>
        requires std::invocable<Func, T&>
        DEVICE void apply_sync(Func&& func) {
            if ( threadIdx.x == owning_thread ) {
                func(wrapped);
            }
            sync();
        }

        DEVICE ~warp_wide() {
            if ( threadIdx.x == owning_thread ) {
                wrapped.~T();
            }
        }
    };

    HOSTDEVICE void swap(swappable auto o1, swappable auto o2) noexcept {
        // static_assert();
        auto tmp { std::move(o1) };
        o1 = std::move(o2);
        o2 = std::move(tmp);
    }

    template<typename T>
    cudaPointerAttributes get_attributes(const T& item) {
        cudaPointerAttributes attr;
        CUDA_CHECK(cudaPointerGetAttributes(&attr, &item));
        return attr;
    }
}  // namespace cuda

#include "undefs.cuh"
#endif
