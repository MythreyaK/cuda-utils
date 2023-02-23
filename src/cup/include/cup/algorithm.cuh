#pragma once

// namespace cup {
//     template <typename T1, typename T2>
//     __device__ auto max(T1 a, T2 b) {
//         return cuda_c_max(a, b);
//     }
// }

namespace cup {
    template<typename T>
    __device__ T clamp(T a, T low, T high) {
        return (a < low) ? low : (a > high ? high : a);
    }

}  // namespace cup
