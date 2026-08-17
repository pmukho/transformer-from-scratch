#include "engine/matmul.hpp"

namespace engine {

// Stage 1: implement a correct naive triple loop (row-major).
// Stage 2: AVX2 + FMA inner loop, parallelized with OpenMP over output rows.
void matmul(float *C, const float *A, const float *B, int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;

      for (int k = 0; k < K; k++) {
        sum += A[(m * K) + k] * B[(k * N) + n];
      }

      C[(m * N) + n] = sum;
    }
  }
}

void matmul_bt(float *C, const float *A, const float *B, int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;

      for (int k = 0; k < K; k++) {
        sum += A[(m * K) + k] * B[(n * K) + k];
      }

      C[(m * N) + n] = sum;
    }
  }
}

} // namespace engine
