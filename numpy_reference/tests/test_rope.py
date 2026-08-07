"""RoPE parity tests against a torch complex-number oracle.

torch's complex-multiply RoPE uses the same interleaved pairing as our impl:
view_as_complex treats consecutive pairs (x[2i], x[2i+1]) as (real, imag), and
multiplying by exp(i * angle) is the 2x2 rotation. So it validates forward
directly, without the rotate_half convention mismatch that HuggingFace would hit.
"""

import numpy as np
import torch

from numpy_reference import rope


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _torch_rope_reference(x, base=10000.0):
    """Rotate interleaved pairs of x via complex multiply. x is (seq, head_dim)."""
    t = torch.from_numpy(x)
    seq, dim = t.shape
    i = torch.arange(dim // 2, dtype=torch.float32)
    theta = base ** (-2.0 * i / dim)
    m = torch.arange(seq, dtype=torch.float32)
    angles = torch.outer(m, theta)  # (seq, dim/2)
    freqs = torch.polar(torch.ones_like(angles), angles)  # complex (seq, dim/2)
    xc = torch.view_as_complex(t.reshape(seq, dim // 2, 2).contiguous())
    return torch.view_as_real(xc * freqs).reshape(seq, dim).numpy()


def test_matches_torch_complex_rope():
    """forward matches the torch complex oracle over a full sequence."""
    seq, dim = 8, 16
    x = _rand(seq, dim, seed=0)
    cos, sin = rope.precompute_cos_sin(seq, dim)

    got = rope.forward(x, cos, sin)
    ref = _torch_rope_reference(x)

    assert np.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_position_offset_matches_full_sequence():
    """Decode path: rotating a single token at position p with the sliced tables
    equals row p of the full-sequence result. This is what the KV cache relies on."""
    seq, dim = 8, 16
    x = _rand(seq, dim, seed=1)
    cos, sin = rope.precompute_cos_sin(seq, dim)

    full = rope.forward(x, cos, sin)
    p = 5
    one = rope.forward(x[p : p + 1], cos[p : p + 1], sin[p : p + 1])

    assert np.allclose(one, full[p : p + 1], atol=1e-6)
