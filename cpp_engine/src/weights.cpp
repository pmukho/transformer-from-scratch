#include "engine/weights.hpp"

#include <cstring>
#include <stdexcept>

#include "engine/npy.hpp"

namespace engine {
namespace {

// Copy a float32 npy array into an owning, aligned Tensor.
Tensor to_tensor(const npy::Array& a) {
  if (a.dtype != "<f4")
    throw std::runtime_error("expected float32 (<f4), got " + a.dtype);
  Tensor t(a.shape);
  std::memcpy(t.data(), a.f32(), a.numel() * sizeof(float));
  return t;
}

}  // namespace

Weights load_model(const std::string& path) {
  auto z = npy::load_npz(path);
  auto get = [&](const std::string& key) -> const npy::Array& {
    auto it = z.find(key);
    if (it == z.end()) throw std::runtime_error("missing key in model.npz: " + key);
    return it->second;
  };

  Weights w;
  Config& c = w.config;
  c.dim = get("config.dim").i32()[0];
  c.n_layers = get("config.n_layers").i32()[0];
  c.n_heads = get("config.n_heads").i32()[0];
  c.kv_heads = get("config.kv_heads").i32()[0];
  c.head_dim = get("config.head_dim").i32()[0];
  c.ffn_hidden = get("config.ffn_hidden").i32()[0];
  c.vocab = get("config.vocab").i32()[0];
  c.rope_base = static_cast<float>(get("config.rope_base").f64()[0]);
  c.eps = static_cast<float>(get("config.eps").f64()[0]);

  w.embedding = to_tensor(get("embedding"));
  w.final_norm = to_tensor(get("final_norm"));

  w.blocks.resize(c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    BlockWeights& b = w.blocks[i];
    b.attn_norm = to_tensor(get(p + "attn_norm"));
    b.wq = to_tensor(get(p + "attn.wq"));
    b.wk = to_tensor(get(p + "attn.wk"));
    b.wv = to_tensor(get(p + "attn.wv"));
    b.wo = to_tensor(get(p + "attn.wo"));
    b.ffn_norm = to_tensor(get(p + "ffn_norm"));
    b.w_gate = to_tensor(get(p + "ffn.w_gate"));
    b.w_up = to_tensor(get(p + "ffn.w_up"));
    b.w_down = to_tensor(get(p + "ffn.w_down"));
  }
  return w;
}

}  // namespace engine
