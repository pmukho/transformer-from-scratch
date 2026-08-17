// Parity test: run the C++ engine against the golden fixtures exported from Phase A.
// Checks prefill logits and the token-by-token decode path (the KV-cache milestone).
//
// Usage: parity_test [weights_dir]   (default ../numpy_reference/weights/tiny)

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "engine/model.hpp"
#include "engine/npy.hpp"

using namespace engine;

static float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

static int argmax(const float* a, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (a[i] > a[best]) best = i;
  return best;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "../numpy_reference/weights/tiny";
  const float TOL = 1e-5f;

  try {
    Weights w = load_model(dir + "/model.npz");
    auto fx = npy::load_npz(dir + "/fixtures.npz");
    const auto& tokens_a = fx.at("tokens");
    const auto& prefill = fx.at("prefill_logits");
    const auto& next = fx.at("next_logits");
    const auto& gen = fx.at("gen_tokens");

    const int prompt_len = static_cast<int>(tokens_a.shape[0]);
    const int max_new = static_cast<int>(gen.shape[0]);
    const int vocab = w.config.vocab;

    std::vector<int> tokens(prompt_len);
    for (int i = 0; i < prompt_len; ++i) tokens[i] = tokens_a.i32()[i];

    Model model(std::move(w), prompt_len + max_new);
    bool ok = true;

    // Prefill: compare logits at every prompt position.
    std::vector<float> logits = model.forward(tokens.data(), prompt_len);
    float dp = max_abs_diff(logits.data(), prefill.f32(),
                            static_cast<size_t>(prompt_len) * vocab);
    std::printf("prefill      : max|diff|=%.3g  %s\n", dp, dp < TOL ? "OK" : "FAIL");
    ok &= dp < TOL;

    // Decode step 0 comes from the prefill's last row.
    const float* last = logits.data() + static_cast<size_t>(prompt_len - 1) * vocab;
    float d0 = max_abs_diff(last, next.f32(), vocab);
    int t0 = argmax(last, vocab);
    bool p0 = d0 < TOL && t0 == gen.i32()[0];
    std::printf("decode[0]    : max|diff|=%.3g argmax=%d expect=%d  %s\n", d0, t0,
                gen.i32()[0], p0 ? "OK" : "FAIL");
    ok &= p0;

    // Subsequent steps: feed the golden tokens one at a time through the cache.
    for (int t = 1; t < max_new; ++t) {
      int in = gen.i32()[t - 1];
      std::vector<float> step = model.forward(&in, 1);
      float dd = max_abs_diff(step.data(), next.f32() + static_cast<size_t>(t) * vocab, vocab);
      int tk = argmax(step.data(), vocab);
      bool pass = dd < TOL && tk == gen.i32()[t];
      std::printf("decode[%d]    : max|diff|=%.3g argmax=%d expect=%d  %s\n", t, dd, tk,
                  gen.i32()[t], pass ? "OK" : "FAIL");
      ok &= pass;
    }

    std::printf("%s\n", ok ? "PARITY PASS" : "PARITY FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
