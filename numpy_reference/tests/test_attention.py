"""Attention parity tests against torch scaled_dot_product_attention as oracle.

torch is a test-only dependency. Tensors are (batch, heads, seq, head_dim), the
layout torch's SDPA expects. Three paths are covered:

  * prefill:            one forward over the whole sequence
  * decode:             feed tokens one at a time, threading the cache
  * split prefill+decode: prefill a chunk, then process the rest with a non-empty
                          cache (this is the case that exercises the mask offset
                          with both past_len > 0 and new_len > 1)

Decode and split check that incremental (cached) processing matches a single
full-sequence forward.
"""

import numpy as np
import torch
import torch.nn.functional as F

from numpy_reference import attention


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _torch_causal_reference(q, k, v):
    """Full causal attention over the whole sequence, shapes (batch, heads, seq, dim)."""
    out = F.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), is_causal=True
    )
    return out.numpy()


def test_prefill_matches_torch():
    """One forward over the whole sequence matches torch causal SDPA."""
    b, h, s, d = 2, 4, 6, 16
    q, k, v = _rand(b, h, s, d, seed=0), _rand(b, h, s, d, seed=1), _rand(b, h, s, d, seed=2)

    out, k_full, v_full = attention.forward(q, k, v)
    ref = _torch_causal_reference(q, k, v)

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)
    # with no past, the returned cache is just the current k/v
    np.testing.assert_allclose(k_full, k)
    np.testing.assert_allclose(v_full, v)


def test_decode_parity_with_prefill():
    """Feeding one token at a time, threading the cache, equals a full prefill."""
    b, h, s, d = 2, 4, 5, 16
    q, k, v = _rand(b, h, s, d, seed=3), _rand(b, h, s, d, seed=4), _rand(b, h, s, d, seed=5)

    past_k = past_v = None
    outs = []
    for t in range(s):
        qt, kt, vt = q[:, :, t : t + 1, :], k[:, :, t : t + 1, :], v[:, :, t : t + 1, :]
        ot, past_k, past_v = attention.forward(qt, kt, vt, past_k, past_v)
        outs.append(ot)
    decode_out = np.concatenate(outs, axis=-2)

    ref = _torch_causal_reference(q, k, v)
    np.testing.assert_allclose(decode_out, ref, atol=1e-5, rtol=1e-5)


def test_split_prefill_then_decode_chunk():
    """Prefill the first p tokens, then process the rest with a non-empty cache."""
    b, h, s, d = 2, 4, 6, 16
    q, k, v = _rand(b, h, s, d, seed=6), _rand(b, h, s, d, seed=7), _rand(b, h, s, d, seed=8)
    p = 4

    out1, pk, pv = attention.forward(q[:, :, :p], k[:, :, :p], v[:, :, :p])
    out2, _, _ = attention.forward(q[:, :, p:], k[:, :, p:], v[:, :, p:], pk, pv)
    combined = np.concatenate([out1, out2], axis=-2)

    ref = _torch_causal_reference(q, k, v)
    np.testing.assert_allclose(combined, ref, atol=1e-5, rtol=1e-5)
