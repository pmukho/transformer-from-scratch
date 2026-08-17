#include "engine/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/matmul.hpp"

namespace engine {

void rmsnorm(float *out, const float *x, const float *weight, int rows, int dim,
             float eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (static_cast<size_t>(r) * dim);
    float *outr = out + (static_cast<size_t>(r) * dim);

    float square_sum = 0.0F;
    for (int d = 0; d < dim; d++)
      square_sum += xr[d] * xr[d];
    float rms = std::sqrt((square_sum / static_cast<float>(dim)) + eps);

    for (int d = 0; d < dim; d++)
      outr[d] = (xr[d] / rms) * weight[d];
  }
}

void rope_precompute(float *cos, float *sin, int max_pos, int head_dim,
                     float base) {
  int half_dim = head_dim / 2;
  for (int m = 0; m < max_pos; m++) {
    for (int i = 0; i < half_dim; i++) {
      double theta = std::pow(static_cast<double>(base), -2.0 * i / head_dim);
      double angle = m * theta;
      cos[(m * half_dim) + i] = std::cos(angle);
      sin[(m * half_dim) + i] = std::sin(angle);
    }
  }
}

void rope_apply(float *x, const float *cos, const float *sin, int rows,
                int n_heads, int head_dim, int pos0) {
  for (int r = 0; r < rows; r++) {
    float *xr = x + (static_cast<size_t>(r) * n_heads * head_dim);

    for (int h = 0; h < n_heads; h++) {
      float *xrh = xr + (static_cast<size_t>(h) * head_dim);

      int half_dim = head_dim / 2;
      for (int i = 0; i < half_dim; i++) {
        float x0 = xrh[2 * i];
        float x1 = xrh[(2 * i) + 1];

        float c = cos[((pos0 + r) * half_dim) + i];
        float s = sin[((pos0 + r) * half_dim) + i];

        xrh[2 * i] = (c * x0) - (s * x1);
        xrh[(2 * i) + 1] = (s * x0) + (c * x1);
      }
    }
  }
}

void softmax_rows(float *x, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float *xr = x + (static_cast<size_t>(r) * cols);

    float max = xr[0];
    for (int c = 1; c < cols; c++) {
      max = std::max(xr[c], max);
    }

    float exp_sum = 0.0F;
    for (int c = 0; c < cols; c++) {
      xr[c] = std::exp(xr[c] - max);
      exp_sum += xr[c];
    }

    for (int c = 0; c < cols; c++) {
      xr[c] = xr[c] / exp_sum;
    }
  }
}

void swiglu_ffn(float *out, const float *x, const BlockWeights &w, int rows,
                int dim, int hidden) {
  std::vector<float> gate(static_cast<size_t>(rows * hidden));
  std::vector<float> up(static_cast<size_t>(rows * hidden));

  matmul(gate.data(), x, w.w_gate.data(), rows, dim, hidden);
  matmul(up.data(), x, w.w_up.data(), rows, dim, hidden);

  for (int r = 0; r < rows; r++) {
    float *gate_r = gate.data() + static_cast<size_t>(r * hidden);
    float *up_r = up.data() + static_cast<size_t>(r * hidden);

    for (int d = 0; d < hidden; d++) {
      float silu = gate_r[d] / (1.0F + std::exp(-gate_r[d]));
      gate_r[d] = silu * up_r[d];
    }
  }

  matmul(out, gate.data(), w.w_down.data(), rows, hidden, dim);
}

void multihead(float *out, const float *x, const BlockWeights &w,
               const Config &cfg, const float *cos, const float *sin,
               int new_len, int pos0, KVCache &cache) {
  int dim = cfg.dim;
  int n_heads = cfg.n_heads;
  int kv_heads = cfg.kv_heads;
  int head_dim = cfg.head_dim;
  int q_dim = n_heads * head_dim;
  int kv_dim = kv_heads * head_dim;
  int total = pos0 + new_len;
  int group_size = n_heads / kv_heads;
  float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> q(static_cast<size_t>(new_len * q_dim));
  std::vector<float> k_new(static_cast<size_t>(new_len * kv_dim));
  std::vector<float> v_new(static_cast<size_t>(new_len * kv_dim));
  std::vector<float> scores(static_cast<size_t>(new_len * total));
  std::vector<float> attn(static_cast<size_t>(new_len * q_dim));

  // Project q/k/v
  matmul(q.data(), x, w.wq.data(), new_len, dim, q_dim);
  matmul(k_new.data(), x, w.wk.data(), new_len, dim, kv_dim);
  matmul(v_new.data(), x, w.wv.data(), new_len, dim, kv_dim);

  // Apply RoPE to q and k
  rope_apply(q.data(), cos, sin, new_len, n_heads, head_dim, pos0);
  rope_apply(k_new.data(), cos, sin, new_len, kv_heads, head_dim, pos0);

  // Append k/v into the cache
  for (int i = 0; i < new_len; i++) {
    std::memcpy(cache.k.data() + (static_cast<size_t>(pos0 + i) * kv_dim),
                k_new.data() + (static_cast<size_t>(i) * kv_dim),
                kv_dim * sizeof(float));
    std::memcpy(cache.v.data() + (static_cast<size_t>(pos0 + i) * kv_dim),
                v_new.data() + (static_cast<size_t>(i) * kv_dim),
                kv_dim * sizeof(float));
  }
  cache.len = total;

  float *k_all = cache.k.data();
  float *v_all = cache.v.data();

  // Run causal attention per head
  for (int h = 0; h < n_heads; h++) {
    int group = h / group_size;
    int q_off = h * head_dim;
    int k_off = group * head_dim;

    for (int i = 0; i < new_len; i++) {
      float *q_i = q.data() + (static_cast<size_t>(i) * q_dim) + q_off;
      int q_pos = pos0 + i;
      float *s_row = scores.data() + (static_cast<size_t>(i) * total);

      for (int j = 0; j < total; j++) {
        if (j > q_pos) {
          s_row[j] = -INFINITY;
        } else {
          float *k_j = k_all + (static_cast<size_t>(j) * kv_dim) + k_off;
          float sum = 0.0F;
          for (int d = 0; d < head_dim; d++)
            sum += q_i[d] * k_j[d];
          s_row[j] = sum * scale;
        }
      }
    }

    softmax_rows(scores.data(), new_len, total);

    for (int i = 0; i < new_len; i++) {
      float *o_i = attn.data() + (static_cast<size_t>(i) * q_dim) + q_off;
      for (int d = 0; d < head_dim; d++)
        o_i[d] = 0.0F;

      float *s_row = scores.data() + (static_cast<size_t>(i) * total);
      for (int j = 0; j < total; j++) {
        float *v_j = v_all + (static_cast<size_t>(j) * kv_dim) + k_off;
        for (int d = 0; d < head_dim; d++)
          o_i[d] += s_row[j] * v_j[d];
      }
    }
  }

  // Merge heads
  matmul(out, attn.data(), w.wo.data(), new_len, q_dim, dim);
}

} // namespace engine
