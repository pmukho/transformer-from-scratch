// Scalar op kernels, mirroring the Phase A modules. Each writes into a
// caller-owned output buffer. The model orchestration (model.cpp) composes
// these; the parity test drives the whole thing against the exported fixtures.

#pragma once

#include "engine/config.hpp"
#include "engine/kv_cache.hpp"
#include "engine/weights.hpp"

namespace engine {

// out[rows,dim] = (x / rms(x)) * weight, normalized over the last axis.
// rms(row) = sqrt(mean(row^2) + eps).
void rmsnorm(float *out, const float *x, const float *weight, int rows, int dim,
             float eps);

// Build interleaved-pair RoPE tables. cos/sin are (max_pos, head_dim/2), where
// entry [m, i] = cos/sin(m * base^(-2i/head_dim)). Compute the angles in double
// and store float32, to match the exporter's tables.
void rope_precompute(float *cos, float *sin, int max_pos, int head_dim,
                     float base);

// Rotate interleaved pairs of x in place. x is (rows, n_heads * head_dim);
// within each head's head_dim block, pair (2i, 2i+1) is rotated by the angle
// for absolute position pos0 + row. Used for q (n_heads) and the new k
// (kv_heads).
void rope_apply(float *x, const float *cos, const float *sin, int rows,
                int n_heads, int head_dim, int pos0);

// In-place softmax over each row of (rows, cols). Numerically stable (subtract
// the row max before exp).
void softmax_rows(float *x, int rows, int cols);

// SwiGLU FFN: out[rows,dim] = ( silu(x @ w_gate) * (x @ w_up) ) @ w_down. No
// bias. silu(z) = z * sigmoid(z). Uses w_gate/w_up (dim, hidden) and w_down
// (hidden, dim).
void swiglu_ffn(float *out, const float *x, const BlockWeights &w, int rows,
                int dim, int hidden);

// Causal multi-head attention with a KV cache, for `new_len` tokens whose first
// absolute position is pos0 (the cache already holds pos0 rows). Steps: project
// q/k/v, apply RoPE to q and the new k, append k/v into the cache, run causal
// attention per head (GQA: query head h uses kv head h / (n_heads/kv_heads)),
// merge heads, apply the output projection. x and out are (new_len, dim).
void multihead(float *out, const float *x, const BlockWeights &w,
               const Config &cfg, const float *cos, const float *sin,
               int new_len, int pos0, KVCache &cache);

} // namespace engine
