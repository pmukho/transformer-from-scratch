// Model weights loaded from model.npz into owning tensors.
//
// Mirrors the nested structure the Phase A exporter flattens: a token embedding
// (also the tied lm_head), one BlockWeights per layer, and a final norm gain.

#pragma once

#include <string>
#include <vector>

#include "engine/config.hpp"
#include "engine/tensor.hpp"

namespace engine {

struct BlockWeights {
  Tensor attn_norm; // (dim,)
  Tensor wq;        // (dim, n_heads * head_dim)
  Tensor wk;        // (dim, kv_heads * head_dim)
  Tensor wv;        // (dim, kv_heads * head_dim)
  Tensor wo;        // (n_heads * head_dim, dim)
  Tensor ffn_norm;  // (dim,)
  Tensor w_gate;    // (dim, ffn_hidden)
  Tensor w_up;      // (dim, ffn_hidden)
  Tensor w_down;    // (ffn_hidden, dim)
};

struct Weights {
  Config config;
  Tensor embedding; // (vocab, dim)
  std::vector<BlockWeights> blocks;
  Tensor final_norm; // (dim,)
};

// Load and validate model.npz (flat keys from scripts/export.py) into Weights.
Weights load_model(const std::string &path);

} // namespace engine
