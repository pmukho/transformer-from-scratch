// Model hyperparameters, loaded from the config.* scalars in model.npz.

#pragma once

namespace engine {

struct Config {
  int dim = 0;
  int n_layers = 0;
  int n_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int ffn_hidden = 0;
  int vocab = 0;
  float rope_base = 0.0F;
  float eps = 0.0F;
};

} // namespace engine
