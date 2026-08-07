"""Decoder block parity tests against a torch reference.

The oracle composes the same pieces the block does: pre-norm RMSNorm, the MHA
reference (matmul projections + interleaved RoPE + causal SDPA + GQA + merge),
residual, pre-norm RMSNorm, SwiGLU, residual. torch is a test-only dependency.

Cases: prefill and token-by-token decode parity (both over MHA and GQA), plus
split prefill + decode. Decode and split cover the KV cache.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from numpy_reference import block, rope


# float64 everywhere: this block stacks two norms, six matmuls, softmax, silu and
# two residuals, and the residual stream grows large. float32 noise then exceeds a
# tight tolerance for reasons that are precision, not correctness. Testing in
# float64 isolates the algorithm; float32 behavior is covered by the op-level tests.
def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def _block_weights(dim, hidden, n_heads, kv_heads, head_dim, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape)

    return {
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


def _torch_rms_norm(x, w, eps=1e-6):
    return F.rms_norm(
        torch.from_numpy(x), (x.shape[-1],), weight=torch.from_numpy(w), eps=eps
    ).numpy()


def _torch_rope(t, base=10000.0):
    b, h, s, d = t.shape
    i = torch.arange(d // 2, dtype=torch.float64)
    theta = base ** (-2.0 * i / d)
    m = torch.arange(s, dtype=torch.float64)
    freqs = torch.polar(torch.ones(s, d // 2, dtype=torch.float64), torch.outer(m, theta))
    tc = torch.view_as_complex(t.reshape(b, h, s, d // 2, 2).contiguous())
    return torch.view_as_real(tc * freqs).reshape(b, h, s, d)


def _torch_mha(h, w, n_heads, kv_heads):
    b, s, _ = h.shape
    head_dim = w["wq"].shape[-1] // n_heads
    th = torch.from_numpy(h)
    wq, wk = torch.from_numpy(w["wq"]), torch.from_numpy(w["wk"])
    wv, wo = torch.from_numpy(w["wv"]), torch.from_numpy(w["wo"])

    def split(proj, heads):
        return (th @ proj).reshape(b, s, heads, head_dim).transpose(1, 2)

    q = _torch_rope(split(wq, n_heads))
    k = _torch_rope(split(wk, kv_heads))
    v = split(wv, kv_heads)
    rep = n_heads // kv_heads
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return (out.transpose(1, 2).reshape(b, s, n_heads * head_dim) @ wo).numpy()


def _torch_swiglu(x, w):
    tx = torch.from_numpy(x)
    hidden = F.silu(tx @ torch.from_numpy(w["w_gate"])) * (tx @ torch.from_numpy(w["w_up"]))
    return (hidden @ torch.from_numpy(w["w_down"])).numpy()


def _torch_block_reference(x, weights, n_heads, kv_heads):
    """Full pre-norm decoder block over the whole sequence, returns (b, s, dim)."""
    h = _torch_rms_norm(x, weights["attn_norm"])
    x1 = x + _torch_mha(h, weights["attn"], n_heads, kv_heads)
    h2 = _torch_rms_norm(x1, weights["ffn_norm"])
    return x1 + _torch_swiglu(h2, weights["ffn"])


# Shared small config
B, S, DIM, N_HEADS, HEAD_DIM, HIDDEN = 2, 6, 32, 4, 8, 64


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_prefill_matches_torch(kv_heads):
    """One forward over the whole sequence matches the torch block reference."""
    x = _rand(B, S, DIM, seed=0)
    weights = _block_weights(DIM, HIDDEN, N_HEADS, kv_heads, HEAD_DIM, seed=1)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    out, _, _ = block.forward(x, weights, cos, sin, n_heads=N_HEADS, kv_heads=kv_heads)
    ref = _torch_block_reference(x, weights, N_HEADS, kv_heads)

    np.testing.assert_allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kv_heads", [N_HEADS, 2])
def test_decode_parity_with_prefill(kv_heads):
    """Token-by-token decode, threading the cache, equals a full prefill."""
    x = _rand(B, S, DIM, seed=2)
    weights = _block_weights(DIM, HIDDEN, N_HEADS, kv_heads, HEAD_DIM, seed=3)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)

    past_k = past_v = None
    outs = []
    for t in range(S):
        ot, past_k, past_v = block.forward(
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

    ref = _torch_block_reference(x, weights, N_HEADS, kv_heads)
    np.testing.assert_allclose(decode_out, ref, atol=1e-6, rtol=1e-6)


def test_split_prefill_then_decode_chunk():
    """Prefill the first p tokens, then process the rest with a non-empty cache."""
    x = _rand(B, S, DIM, seed=4)
    weights = _block_weights(DIM, HIDDEN, N_HEADS, N_HEADS, HEAD_DIM, seed=5)
    cos, sin = rope.precompute_cos_sin(S, HEAD_DIM)
    p = 4

    out1, pk, pv = block.forward(x[:, :p, :], weights, cos, sin, n_heads=N_HEADS, kv_heads=N_HEADS)
    out2, _, _ = block.forward(
        x[:, p:, :], weights, cos, sin, pk, pv, n_heads=N_HEADS, kv_heads=N_HEADS
    )
    combined = np.concatenate([out1, out2], axis=1)

    ref = _torch_block_reference(x, weights, N_HEADS, N_HEADS)
    np.testing.assert_allclose(combined, ref, atol=1e-6, rtol=1e-6)
