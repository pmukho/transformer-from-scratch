// Per-layer KV cache: preallocated key/value buffers, no allocation in the hot
// path.

#pragma once

#include "engine/tensor.hpp"

namespace engine {

struct KVCache {
  Tensor k;    // (max_seq, kv_heads * head_dim), row-major
  Tensor v;    // (max_seq, kv_heads * head_dim)
  int len = 0; // number of valid rows currently cached
  int max_seq = 0;
  int kv_dim = 0;

  KVCache() = default;
  KVCache(int max_seq_, int kv_dim_)
      : k(Tensor::zeros(
            {static_cast<size_t>(max_seq_), static_cast<size_t>(kv_dim_)})),
        v(Tensor::zeros(
            {static_cast<size_t>(max_seq_), static_cast<size_t>(kv_dim_)})),
        max_seq(max_seq_), kv_dim(kv_dim_) {}

  void reset() { len = 0; }
};

} // namespace engine
