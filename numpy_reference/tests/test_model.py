"""Full-model parity tests against a torch reference.

The oracle reuses the decoder-block reference from test_block and wraps it with
the embedding lookup, final RMSNorm, and weight-tied lm_head. Run in float64 (the
stack is deep and the residual stream grows large), so tolerances stay tight and
measure correctness, not float32 noise.

Cases: prefill, token-by-token decode parity, and split prefill + decode. The
decode case is the Milestone A KV-cache check at the whole-model level.
"""

import numpy as np
import pytest
from test_block import _torch_block_reference, _torch_rms_norm

from numpy_reference import model, rope


def _model_weights(vocab, dim, hidden, n_layers, n_heads, kv_heads, head_dim, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape)

    blocks = []
    for _ in range(n_layers):
        blocks.append(
            {
                "attn_norm": r(dim),
                "attn": {
                    "wq": r(dim, n_heads * head_dim),
                    "wk": r(dim, kv_heads * head_dim),
                    "wv": r(dim, kv_heads * head_dim),
                    "wo": r(n_heads * head_dim, dim),
                },
                "ffn_norm": r(dim),
                "ffn": {
                    "w_gate": r(dim, hidden),
                    "w_up": r(dim, hidden),
                    "w_down": r(hidden, dim),
                },
            }
        )
    return {"embedding": r(vocab, dim), "blocks": blocks, "final_norm": r(dim)}


def _torch_model_reference(tokens, weights, n_heads, kv_heads):
    """Full stack over the whole sequence, returns logits (batch, seq, vocab)."""
    emb = weights["embedding"]
    x = emb[tokens]  # (batch, seq, dim)
    for bw in weights["blocks"]:
        x = _torch_block_reference(x, bw, n_heads, kv_heads)
    x = _torch_rms_norm(x, weights["final_norm"])
    return x @ emb.T


# Shared small config (mirrors the CLAUDE.md example: small and random-init)
B, S, VOCAB, DIM, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 2, 6, 32, 32, 2, 4, 8, 64


def _tokens(seed=0):
    return np.random.default_rng(seed).integers(0, VOCAB, size=(B, S))


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_prefill_matches_torch(kv_heads):
    """One forward over the whole sequence matches the torch model reference."""
    tokens = _tokens(seed=0)
    weights = _model_weights(VOCAB, DIM, HIDDEN, N_LAYERS, N_HEADS, kv_heads, HEAD_DIM, seed=1)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    logits, _ = model.forward(tokens, weights, cos, sin, n_heads=N_HEADS, kv_heads=kv_heads)
    ref = _torch_model_reference(tokens, weights, N_HEADS, kv_heads)

    np.testing.assert_allclose(logits, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_decode_parity_with_prefill(kv_heads):
    """Token-by-token decode, threading the per-layer cache, equals a full prefill."""
    tokens = _tokens(seed=2)
    weights = _model_weights(VOCAB, DIM, HIDDEN, N_LAYERS, N_HEADS, kv_heads, HEAD_DIM, seed=3)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    past = None
    logit_steps = []
    for t in range(S):
        lt, past = model.forward(
            tokens[:, t : t + 1], weights, cos, sin, past, n_heads=N_HEADS, kv_heads=kv_heads
        )
        logit_steps.append(lt)
    decode_logits = np.concatenate(logit_steps, axis=1)

    ref = _torch_model_reference(tokens, weights, N_HEADS, kv_heads)
    np.testing.assert_allclose(decode_logits, ref, atol=1e-6, rtol=1e-6)


def test_split_prefill_then_decode_chunk():
    """Prefill the first p tokens, then process the rest with a non-empty cache."""
    tokens = _tokens(seed=4)
    weights = _model_weights(VOCAB, DIM, HIDDEN, N_LAYERS, N_HEADS, N_HEADS, HEAD_DIM, seed=5)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)
    p = 4

    logits1, past = model.forward(
        tokens[:, :p], weights, cos, sin, n_heads=N_HEADS, kv_heads=N_HEADS
    )
    logits2, _ = model.forward(
        tokens[:, p:], weights, cos, sin, past, n_heads=N_HEADS, kv_heads=N_HEADS
    )
    combined = np.concatenate([logits1, logits2], axis=1)

    ref = _torch_model_reference(tokens, weights, N_HEADS, N_HEADS)
    np.testing.assert_allclose(combined, ref, atol=1e-6, rtol=1e-6)
