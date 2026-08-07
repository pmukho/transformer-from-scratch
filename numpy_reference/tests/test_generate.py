"""Greedy generation tests.

The core check: greedy decode with the KV cache must produce the exact same token
sequence as greedy decode that re-forwards the whole prefix every step. Tokens are
integer argmax, so this is exact equality, no tolerance. Run in float64 so a tiny
float32 difference can never flip an argmax at a near-tie.

Reuses the model weight builder from test_model.
"""

import numpy as np
import pytest
from test_model import _model_weights

from numpy_reference import generate, model, rope

B, PROMPT, VOCAB, DIM, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 2, 4, 32, 32, 2, 4, 8, 64
MAX_NEW = 5


def _greedy_reforward(tokens, weights, cos, sin, max_new, n_heads, kv_heads):
    """Reference greedy decode with no cache: re-run the full sequence each step."""
    seq = tokens.copy()
    generated = []
    for _ in range(max_new):
        logits, _ = model.forward(seq, weights, cos, sin, n_heads=n_heads, kv_heads=kv_heads)
        nxt = np.argmax(logits[:, -1, :], axis=-1)
        generated.append(nxt)
        seq = np.concatenate([seq, nxt[:, None]], axis=1)
    return np.stack(generated, axis=1)


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_greedy_cache_matches_full_reforward(kv_heads):
    """Cached greedy decode equals the no-cache re-forward, token for token."""
    tokens = np.random.default_rng(0).integers(0, VOCAB, size=(B, PROMPT))
    weights = _model_weights(VOCAB, DIM, HIDDEN, N_LAYERS, N_HEADS, kv_heads, HEAD_DIM, seed=1)
    cos, sin = rope.precompute_cos_sin(PROMPT + MAX_NEW, HEAD_DIM)

    got = generate.generate(tokens, weights, cos, sin, MAX_NEW, n_heads=N_HEADS, kv_heads=kv_heads)
    ref = _greedy_reforward(tokens, weights, cos, sin, MAX_NEW, N_HEADS, kv_heads)

    assert got.shape == (B, MAX_NEW)
    assert np.array_equal(got, ref)


def test_generation_is_deterministic():
    """Greedy decode has no randomness: two runs give the identical sequence."""
    tokens = np.random.default_rng(2).integers(0, VOCAB, size=(B, PROMPT))
    weights = _model_weights(VOCAB, DIM, HIDDEN, N_LAYERS, N_HEADS, N_HEADS, HEAD_DIM, seed=3)
    cos, sin = rope.precompute_cos_sin(PROMPT + MAX_NEW, HEAD_DIM)

    a = generate.generate(tokens, weights, cos, sin, MAX_NEW, n_heads=N_HEADS)
    b = generate.generate(tokens, weights, cos, sin, MAX_NEW, n_heads=N_HEADS)

    assert np.array_equal(a, b)
