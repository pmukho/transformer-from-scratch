#include "engine/matmul.hpp"

#include <cstring>

#ifdef ENGINE_AVX2
#include <immintrin.h>
#endif

namespace engine {

// C[M,N] = A[M,K] @ B[K,N], row-major.
//   naive: scalar triple loop.
//   avx2:  vectorize along N. For each row m, loop k, broadcast A[m,k] and FMA
//          it across contiguous 8-float chunks of B[k, n..] into the C[m, n..]
//          accumulators (SAXPY form). Parallelize the m loop with OpenMP.
void matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef ENGINE_AVX2
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    std::memset(&C[m * N], 0, N * sizeof(float));

    for (int k = 0; k < K; k++) {
      __m256 a_mk = _mm256_set1_ps(A[(m * K) + k]);

      int n = 0;
      // AVX loop
      for (; n + 8 <= N; n += 8) {
        __m256 b_kn = _mm256_loadu_ps(&B[(k * N) + n]);
        __m256 c_mn = _mm256_loadu_ps(&C[(m * N) + n]);
        _mm256_storeu_ps(&C[(m * N) + n], _mm256_fmadd_ps(a_mk, b_kn, c_mn));
      }
      // Scalar tail
      for (; n < N; n++) {
        C[(m * N) + n] += A[(m * K) + k] * B[(k * N) + n];
      }
    }
  }

#else
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;

      for (int k = 0; k < K; k++) {
        sum += A[(m * K) + k] * B[(k * N) + n];
      }

      C[(m * N) + n] = sum;
    }
  }
#endif
}

#ifdef ENGINE_AVX2
// Horizontal sum of a __m256 using reductions.
float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);

  __m128 shuf = _mm_movehdup_ps(sum4); // [a, b, c, d] -> [b, b, d, d]
  __m128 sums2 = _mm_add_ps(sum4, shuf);

  shuf = _mm_movehl_ps(shuf, sums2); // [b, b, d, d] -> [d, d, d, d]
  __m128 sum1 = _mm_add_ss(sums2, shuf);

  return _mm_cvtss_f32(sum1);
}
#endif

// C[M,N] = A[M,K] @ B[N,K]^T, with B stored row-major as (N,K).
//   naive: scalar triple loop.
//   avx2:  A[m,:] and B[n,:] are both contiguous over k, so this is a real dot
//          product. Keep a __m256 accumulator, FMA down k in steps of 8, then
//          horizontal-sum + scalar tail for K % 8. Parallelize the m loop.
void matmul_bt(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef ENGINE_AVX2
#pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int k = 0;

      // AVX loop
      __m256 acc = _mm256_setzero_ps();
      for (; k + 8 <= K; k += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(&A[(m * K) + k]),
                              _mm256_loadu_ps(&B[(n * K) + k]), acc);
      }
      float sum = hsum(acc);

      // Scalar tail
      for (; k < K; k++) {
        sum += A[(m * K) + k] * B[(n * K) + k];
      }
      C[(m * N) + n] = sum;
    }
  }
#else
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;

      for (int k = 0; k < K; k++) {
        sum += A[(m * K) + k] * B[(n * K) + k];
      }

      C[(m * N) + n] = sum;
    }
  }
#endif
}

} // namespace engine
