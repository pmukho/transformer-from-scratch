"""Rotary positional embedding (RoPE).

Position is encoded by rotating each adjacent pair of features by an angle that
grows with the position index. For pair index i in [0, head_dim/2) the frequency
is

    theta_i = base ** (-2i / head_dim)

and at position m the rotation angle is m * theta_i. Each pair (x_2i, x_2i+1) is
rotated by that angle:

    x'_2i   = x_2i * cos - x_2i+1 * sin
    x'_2i+1 = x_2i * sin + x_2i+1 * cos

Same frequency construction as sinusoidal encoding, applied as a rotation of Q/K
instead of an additive term.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def precompute_cos_sin(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build the cos/sin tables for positions 0..seq_len-1.

    Args:
        seq_len: Number of positions to precompute.
        head_dim: Per-head feature size. Must be even (features come in pairs).
        base: Frequency base (theta), 10000 in Llama.

    Returns:
        cos, sin: each shape (seq_len, head_dim/2), where entry [m, i] is the
        cos/sin of angle m * theta_i.
    """
    i = np.arange(head_dim // 2)
    theta = base ** (-2 * i / head_dim)
    m = np.arange(seq_len)
    angles = np.outer(m, theta)
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def forward(
    x: NDArray[np.floating],
    cos: NDArray[np.floating],
    sin: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Rotate the adjacent feature pairs of x by the precomputed angles.

    Call once for Q and once for K. During KV-cache decode, slice cos/sin to the
    positions of the tokens in x so each token gets its own angle.

    Args:
        x: Input with shape (..., seq_len, head_dim). Pairs are (x[..., 2i],
            x[..., 2i+1]).
        cos: cos table sliced to the positions of x, shape (seq_len, head_dim/2).
        sin: sin table sliced to the positions of x, shape (seq_len, head_dim/2).

    Returns:
        x with each pair rotated, same shape as x.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated1 = x1 * cos - x2 * sin
    x_rotated2 = x1 * sin + x2 * cos
    x_out = np.empty_like(x)
    x_out[..., ::2] = x_rotated1
    x_out[..., 1::2] = x_rotated2
    return x_out
