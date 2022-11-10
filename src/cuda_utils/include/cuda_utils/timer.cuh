#ifndef CUDA_TIMER_HPP
#define CUDA_TIMER_HPP

namespace cuda {

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
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_taken, start, stop);
            m_callback(time_taken);
            // out_time = 1000;
        }
    };

    template<typename T>
    float time_it(const T& callback) {
        cudaEvent_t start, stop;
        float       time_taken {};

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        callback();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_taken, start, stop);
        return time_taken;
    };
}  // namespace cuda

#endif
