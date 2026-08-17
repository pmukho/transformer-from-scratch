// Loads model.npz and prints the config plus exact first-5 raw values and shapes
// for a few tensors, so the C++ loader can be diffed against NumPy.

#include <cstdio>
#include <string>

#include "engine/weights.hpp"

using namespace engine;

static void dump(const char* name, const Tensor& t) {
  std::printf("%-24s shape=[", name);
  for (size_t i = 0; i < t.ndim(); ++i)
    std::printf("%zu%s", t.size(i), i + 1 < t.ndim() ? "," : "");
  std::printf("] first5=");
  size_t n = t.numel() < 5 ? t.numel() : 5;
  for (size_t i = 0; i < n; ++i) std::printf("%.8g ", t.data()[i]);
  std::printf("\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: load_check <model.npz>\n");
    return 1;
  }
  Weights w = load_model(argv[1]);
  const Config& c = w.config;
  std::printf(
      "config dim=%d n_layers=%d n_heads=%d kv_heads=%d head_dim=%d "
      "ffn_hidden=%d vocab=%d rope_base=%g eps=%g\n",
      c.dim, c.n_layers, c.n_heads, c.kv_heads, c.head_dim, c.ffn_hidden, c.vocab,
      c.rope_base, c.eps);
  dump("embedding", w.embedding);
  dump("final_norm", w.final_norm);
  dump("blocks[0].attn.wq", w.blocks.front().wq);
  dump("blocks[last].ffn.w_down", w.blocks.back().w_down);

  // Full-model checksum over every weight element (double accumulation).
  double total = 0.0;
  size_t count = 0;
  auto acc = [&](const Tensor& t) {
    for (size_t i = 0; i < t.numel(); ++i) total += t.data()[i];
    count += t.numel();
  };
  acc(w.embedding);
  acc(w.final_norm);
  for (const auto& b : w.blocks) {
    for (const Tensor* t : {&b.attn_norm, &b.wq, &b.wk, &b.wv, &b.wo, &b.ffn_norm,
                            &b.w_gate, &b.w_up, &b.w_down})
      acc(*t);
  }
  std::printf("ALL weights: count=%zu sum=%.10g\n", count, total);
  return 0;
}
