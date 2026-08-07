"""Multi-head attention parity tests against a torch reference.

The oracle mirrors the module's pipeline exactly: matmul projections (no bias),
interleaved complex RoPE on Q/K, causal SDPA per head, GQA head repeat, merge,
output projection. torch's nn.MultiheadAttention doesn't fit as an oracle: it
packs projections differently, adds bias, and has no RoPE or GQA.

Cases: prefill, token-by-token decode parity (parametrized over MHA and GQA),
and split prefill + decode. Decode and split cover the KV cache.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numpy_reference import multihead, rope


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _weights(dim, n_heads, kv_heads, head_dim, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "wq": r(dim, n_heads * head_dim),
        "wk": r(dim, kv_heads * head_dim),
        "wv": r(dim, kv_heads * head_dim),
        "wo": r(n_heads * head_dim, dim),
    }


def _torch_rope(t, base=10000.0):
    """Interleaved RoPE on t of shape (batch, heads, seq, head_dim)."""
    b, h, s, d = t.shape
    i = torch.arange(d // 2, dtype=torch.float32)
    theta = base ** (-2.0 * i / d)
    m = torch.arange(s, dtype=torch.float32)
    freqs = torch.polar(torch.ones(s, d // 2), torch.outer(m, theta))  # (s, d/2) complex
    tc = torch.view_as_complex(t.reshape(b, h, s, d // 2, 2).contiguous())
    return torch.view_as_real(tc * freqs).reshape(b, h, s, d)


def _torch_mha_reference(x, weights, n_heads, kv_heads):
    """Full causal multi-head attention over the whole sequence, returns (b, s, dim)."""
    b, s, dim = x.shape
    head_dim = weights["wq"].shape[-1] // n_heads
    tx = torch.from_numpy(x)
    wq, wk = torch.from_numpy(weights["wq"]), torch.from_numpy(weights["wk"])
    wv, wo = torch.from_numpy(weights["wv"]), torch.from_numpy(weights["wo"])

    def split(proj, heads):
        return (tx @ proj).reshape(b, s, heads, head_dim).transpose(1, 2)

    q = _torch_rope(split(wq, n_heads))
    k = _torch_rope(split(wk, kv_heads))
    v = split(wv, kv_heads)

    rep = n_heads // kv_heads
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (b, heads, s, head_dim)
    out = out.transpose(1, 2).reshape(b, s, n_heads * head_dim) @ wo
    return out.numpy()


# Shared small config
B, S, DIM, N_HEADS, HEAD_DIM = 2, 6, 32, 4, 8


def test_prefill_matches_torch():
    """One forward over the whole sequence matches the torch reference (MHA)."""
    x = _rand(B, S, DIM, seed=0)
    weights = _weights(DIM, N_HEADS, N_HEADS, HEAD_DIM, seed=1)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    out, _, _ = multihead.forward(x, weights, cos, sin, n_heads=N_HEADS, kv_heads=N_HEADS)
    ref = _torch_mha_reference(x, weights, N_HEADS, N_HEADS)

    # 1e-4: float32 through 4 matmuls + softmax vs torch's op ordering (not a logic tolerance)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_gqa_prefill_matches_torch():
    """GQA (kv_heads < n_heads) prefill matches the torch reference."""
    kv_heads = 2
    x = _rand(B, S, DIM, seed=2)
    weights = _weights(DIM, N_HEADS, kv_heads, HEAD_DIM, seed=3)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    out, _, _ = multihead.forward(x, weights, cos, sin, n_heads=N_HEADS, kv_heads=kv_heads)
    ref = _torch_mha_reference(x, weights, N_HEADS, kv_heads)

    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_decode_parity_with_prefill(kv_heads):
    """Token-by-token decode, threading the cache, equals a full prefill (MHA and GQA)."""
    x = _rand(B, S, DIM, seed=4)
    weights = _weights(DIM, N_HEADS, kv_heads, HEAD_DIM, seed=5)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    past_k = past_v = None
    outs = []
    for t in range(S):
        ot, past_k, past_v = multihead.forward(
            x[:, t : t + 1, :],
            weights,
            cos,
            sin,
            past_k,
            past_v,
            n_heads=N_HEADS,
            kv_heads=kv_heads,
        )
        outs.append(ot)
    decode_out = np.concatenate(outs, axis=1)

    ref = _torch_mha_reference(x, weights, N_HEADS, kv_heads)
    np.testing.assert_allclose(decode_out, ref, atol=1e-4, rtol=1e-4)


def test_split_prefill_then_decode_chunk():
    """Prefill the first p tokens, then process the rest with a non-empty cache."""
    x = _rand(B, S, DIM, seed=6)
    weights = _weights(DIM, N_HEADS, N_HEADS, HEAD_DIM, seed=7)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)
    p = 4

    out1, pk, pv = multihead.forward(
        x[:, :p, :], weights, cos, sin, n_heads=N_HEADS, kv_heads=N_HEADS
    )
    out2, _, _ = multihead.forward(
        x[:, p:, :], weights, cos, sin, pk, pv, n_heads=N_HEADS, kv_heads=N_HEADS
    )
    combined = np.concatenate([out1, out2], axis=1)

    ref = _torch_mha_reference(x, weights, N_HEADS, N_HEADS)
    np.testing.assert_allclose(combined, ref, atol=1e-4, rtol=1e-4)
