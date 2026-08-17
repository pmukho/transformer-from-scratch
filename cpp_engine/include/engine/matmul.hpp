// Matmul kernels. Row-major, float32. The Stage 1 job is a correct naive
// implementation; Stage 2 replaces the inner loop with AVX2 + FMA and
// parallelizes with OpenMP. Both projections and the lm_head route through
// these.

#pragma once

namespace engine {

// C[M,N] = A[M,K] @ B[K,N]. Used for the q/k/v/out projections and the FFN.
void matmul(float *C, const float *A, const float *B, int M, int K, int N);

// C[M,N] = A[M,K] @ B[N,K]^T, with B stored row-major as (N,K). Used for the
// attention scores (q @ k^T) and the weight-tied lm_head (x @ embedding^T).
void matmul_bt(float *C, const float *A, const float *B, int M, int K, int N);

} // namespace engine
