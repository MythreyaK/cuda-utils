#ifndef CUP_ALGORITHM_UTILS_CUH
#define CUP_ALGORITHM_UTILS_CUH

#include "defs.cuh"

// namespace cup {
//     template <typename T1, typename T2>
//     __device__ auto max(T1 a, T2 b) {
//         return cuda_c_max(a, b);
//     }
// }

namespace cup {

    template<typename T>
    FORCEINLINE HOSTDEVICE constexpr const T& clamp(const T& a,
                                                    const T& low,
                                                    const T& high) {
        return (a < low) ? low : (a > high ? high : a);
    }

}  // namespace cup

namespace {
    static_assert(cup::clamp(50, 30, 80) == 50);
    static_assert(cup::clamp(10, 30, 80) == 30);
    static_assert(cup::clamp(90, 30, 80) == 80);
    static_assert(cup::clamp(30, 30, 80) == 30);
    static_assert(cup::clamp(80, 30, 80) == 80);

    static_assert(cup::clamp(50.0, 30.0, 80.0) == 50.0);
    static_assert(cup::clamp(10.0, 30.0, 80.0) == 30.0);
    static_assert(cup::clamp(90.0, 30.0, 80.0) == 80.0);
    static_assert(cup::clamp(30.0, 30.0, 80.0) == 30.0);
    static_assert(cup::clamp(80.0, 30.0, 80.0) == 80.0);

    static_assert(cup::clamp(50.0f, 30.0f, 80.0f) == 50.0f);
    static_assert(cup::clamp(10.0f, 30.0f, 80.0f) == 30.0f);
    static_assert(cup::clamp(90.0f, 30.0f, 80.0f) == 80.0f);
    static_assert(cup::clamp(30.0f, 30.0f, 80.0f) == 30.0f);
    static_assert(cup::clamp(80.0f, 30.0f, 80.0f) == 80.0f);
}  // namespace

#include "undefs.cuh"

#endif
