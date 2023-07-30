#pragma once

#include <chrono>

#include "defs.cuh"

namespace cup {

    template<typename T>
    struct scoped_timer {
      private:
        cudaEvent_t start, stop;
        const T&    m_callback {};
        float       time_taken {};

      public:
        scoped_timer(const T& callback) : m_callback(m_callback) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
        }

        ~scoped_timer() {
            // call the callback
            using milli_t = std::chrono::duration<float, std::milli>;
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_taken, start, stop);
            m_callback(milli_t { time_taken });
            // out_time = 1000;
        }
    };

    template<typename T>
    auto time_it(const T& callback) {
        cudaEvent_t start, stop;
        float       time_taken {};

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        callback();

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time_taken, start, stop));
        return std::chrono::duration<float, std::milli> { time_taken };
    };
}  // namespace cup

#include "undefs.cuh"
