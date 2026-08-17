#include "engine/model.hpp"

#include <cstring>
#include <utility>

#include "engine/matmul.hpp"
#include "engine/ops.hpp"

namespace engine {

Model::Model(Weights weights, int max_seq)
    : w_(std::move(weights)), cfg_(w_.config), max_seq_(max_seq) {
  const int kv_dim = cfg_.kv_heads * cfg_.head_dim;
  caches_.reserve(cfg_.n_layers);
  for (int i = 0; i < cfg_.n_layers; ++i)
    caches_.emplace_back(max_seq_, kv_dim);

  const size_t half = static_cast<size_t>(cfg_.head_dim / 2);
  cos_ = Tensor({static_cast<size_t>(max_seq_), half});
  sin_ = Tensor({static_cast<size_t>(max_seq_), half});
  rope_precompute(cos_.data(), sin_.data(), max_seq_, cfg_.head_dim,
                  cfg_.rope_base);
}

void Model::reset() {
  pos_ = 0;
  for (auto &c : caches_)
    c.reset();
}

std::vector<float> Model::forward(const int *tokens, int n) {
  const int dim = cfg_.dim;
  const int hidden = cfg_.ffn_hidden;
  const int vocab = cfg_.vocab;

  // Token embedding: gather rows of the embedding matrix.
  Tensor x({static_cast<size_t>(n), static_cast<size_t>(dim)});
  const float *emb = w_.embedding.data();
  for (int i = 0; i < n; ++i)
    std::memcpy(x.data() + static_cast<size_t>(i) * dim,
                emb + static_cast<size_t>(tokens[i]) * dim,
                dim * sizeof(float));

  Tensor h({static_cast<size_t>(n), static_cast<size_t>(dim)});
  Tensor tmp({static_cast<size_t>(n), static_cast<size_t>(dim)});

  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    const BlockWeights &bw = w_.blocks[layer];

    // Attention sublayer (pre-norm, residual on the un-normalized x).
    rmsnorm(h.data(), x.data(), bw.attn_norm.data(), n, dim, cfg_.eps);
    multihead(tmp.data(), h.data(), bw, cfg_, cos_.data(), sin_.data(), n, pos_,
              caches_[layer]);
    for (int i = 0; i < n * dim; ++i)
      x.data()[i] += tmp.data()[i];

    // Feed-forward sublayer.
    rmsnorm(h.data(), x.data(), bw.ffn_norm.data(), n, dim, cfg_.eps);
    swiglu_ffn(tmp.data(), h.data(), bw, n, dim, hidden);
    for (int i = 0; i < n * dim; ++i)
      x.data()[i] += tmp.data()[i];
  }

  rmsnorm(h.data(), x.data(), w_.final_norm.data(), n, dim, cfg_.eps);

  // Weight-tied lm_head: logits (n, vocab) = h @ embedding^T.
  std::vector<float> logits(static_cast<size_t>(n) * vocab);
  matmul_bt(logits.data(), h.data(), emb, n, dim, vocab);

  pos_ += n;
  return logits;
}

} // namespace engine
