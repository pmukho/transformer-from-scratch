"""Decoder block (pre-norm).

One Llama-style transformer layer:

    h = x + attn(rmsnorm(x))
    y = h + swiglu(rmsnorm(h))

Pre-norm means each sublayer sees a normalized input, but the residual adds the
un-normalized value from before the norm. The block is shape-preserving
(dim in, dim out) so blocks stack. The KV cache is threaded through the attention
sublayer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_reference import multihead, rmsnorm, swiglu


def forward(
    x: NDArray[np.floating],
    weights: dict,
    cos: NDArray[np.floating],
    sin: NDArray[np.floating],
    past_k: NDArray[np.floating] | None = None,
    past_v: NDArray[np.floating] | None = None,
    *,
    n_heads: int,
    kv_heads: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Run one decoder block over x.

    Args:
        x: Input activations, shape (batch, seq, dim).
        weights: Nested dict:
            'attn_norm' (dim,)        RMSNorm gain before attention
            'attn'      dict          passed to multihead.forward (wq, wk, wv, wo)
            'ffn_norm'  (dim,)        RMSNorm gain before the feed-forward
            'ffn'       dict          passed to swiglu.forward (w_gate, w_up, w_down)
        cos, sin: RoPE tables, passed through to multihead.
        past_k, past_v: Attention KV cache, or None on prefill.
        n_heads, kv_heads: Head config, passed through to multihead.

    Returns:
        y: Output activations, shape (batch, seq, dim).
        k_full, v_full: Updated cache from the attention sublayer.
    """
    attn, k_full, v_full = multihead.forward(
        rmsnorm.forward(x, weights["attn_norm"]),
        weights["attn"],
        cos,
        sin,
        past_k,
        past_v,
        n_heads=n_heads,
        kv_heads=kv_heads,
    )

    h = x + attn

    y = h + swiglu.forward(
        rmsnorm.forward(h, weights["ffn_norm"]),
        weights["ffn"],
    )

    return y, k_full, v_full
