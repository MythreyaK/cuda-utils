#ifndef CUDA_CPP_ALGORITHM_HPP
#define CUDA_CPP_ALGORITHM_HPP

// namespace cuda {
//     template <typename T1, typename T2>
//     __device__ auto max(T1 a, T2 b) {
//         return cuda_c_max(a, b);
//     }
// }

namespace cuda {
    template<typename T>
    __device__ T clamp(T a, T low, T high) {
        return (a < low) ? low : (a > high ? high : a);
    }

}  // namespace cuda

#endif
