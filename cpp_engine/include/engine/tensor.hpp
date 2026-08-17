// Rank-N float32 tensor: shape, row-major strides, raw data pointer.
//
// Two modes: an owning tensor with a 32-byte aligned buffer (for AVX2 loads),
// or a non-owning view over external memory (e.g. a slice of the KV cache).
// Move-only so ownership is never ambiguous. Preallocate these outside hot
// loops; the ops write into already-allocated output tensors.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace engine {

constexpr size_t kAlign = 32; // AVX2 register width in bytes

inline float *aligned_alloc_f32(size_t n) {
  size_t bytes = ((n * sizeof(float) + kAlign - 1) / kAlign) * kAlign;
  void *p = std::aligned_alloc(kAlign, bytes);
  if (!p)
    throw std::bad_alloc();
  return static_cast<float *>(p);
}

class Tensor {
public:
  Tensor() = default;

  // Owning tensor with a fresh aligned buffer (contents uninitialized).
  explicit Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    compute_strides();
    n_ = numel();
    data_ = aligned_alloc_f32(n_);
    owns_ = true;
  }

  // Non-owning view over external data.
  Tensor(std::vector<size_t> shape, float *data)
      : shape_(std::move(shape)), data_(data) {
    compute_strides();
    n_ = numel();
  }

  ~Tensor() {
    if (owns_)
      std::free(data_);
  }

  Tensor(Tensor &&o) noexcept { move_from(o); }
  Tensor &operator=(Tensor &&o) noexcept {
    if (this != &o) {
      if (owns_)
        std::free(data_);
      move_from(o);
    }
    return *this;
  }
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  static Tensor zeros(std::vector<size_t> shape) {
    Tensor t(std::move(shape));
    std::fill(t.data_, t.data_ + t.n_, 0.0f);
    return t;
  }

  float *data() { return data_; }
  const float *data() const { return data_; }
  size_t numel() const {
    size_t n = 1;
    for (size_t d : shape_)
      n *= d;
    return n;
  }
  size_t ndim() const { return shape_.size(); }
  size_t size(size_t d) const { return shape_[d]; }
  const std::vector<size_t> &shape() const { return shape_; }
  const std::vector<size_t> &strides() const { return strides_; }

  float &operator[](size_t i) { return data_[i]; }
  float operator[](size_t i) const { return data_[i]; }

private:
  void compute_strides() {
    strides_.assign(shape_.size(), 1);
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
      strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
  void move_from(Tensor &o) {
    shape_ = std::move(o.shape_);
    strides_ = std::move(o.strides_);
    data_ = o.data_;
    n_ = o.n_;
    owns_ = o.owns_;
    o.data_ = nullptr;
    o.owns_ = false;
    o.n_ = 0;
  }

  std::vector<size_t> shape_, strides_;
  float *data_ = nullptr;
  size_t n_ = 0;
  bool owns_ = false;
};

} // namespace engine
