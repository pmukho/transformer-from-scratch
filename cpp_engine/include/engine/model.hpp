// Full decoder stack. Owns the weights, per-layer KV caches, and RoPE tables,
// and tracks the current absolute position so prefill and incremental decode
// share one code path.

#pragma once

#include <vector>

#include "engine/config.hpp"
#include "engine/kv_cache.hpp"
#include "engine/tensor.hpp"
#include "engine/weights.hpp"

namespace engine {

class Model {
public:
  // max_seq bounds the KV cache and RoPE tables (prompt length + tokens to
  // generate).
  Model(Weights weights, int max_seq);

  // Process n tokens at the current position, advancing the caches. Returns
  // logits row-major (n, vocab). n > 1 is prefill; n == 1 is a decode step.
  std::vector<float> forward(const int *tokens, int n);

  // Clear caches and rewind to position 0.
  void reset();

  const Config &config() const { return cfg_; }

private:
  Weights w_;
  Config cfg_;
  int max_seq_;
  int pos_ = 0;
  std::vector<KVCache> caches_;
  Tensor cos_;
  Tensor sin_;
};

} // namespace engine
