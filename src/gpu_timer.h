//
// Timer.
//

#include <cuda_runtime.h>

namespace svs {

struct GpuTimer {
  cudaEvent_t event_start;
  cudaEvent_t event_stop;

  GpuTimer() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
  }

  void start() { cudaEventRecord(event_start, 0); }

  void stop() { cudaEventRecord(event_stop, 0); }

  float elapsed() {
    float elapsed;
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsed, event_start, event_stop);
    return elapsed;
  }
};

}  // namespace svs
