"""Multi-head attention block.

Projects the input into per-head queries, keys, and values, applies RoPE to Q and
K, runs causal KV-cache-aware attention per head, merges the heads, and applies
the output projection. No bias terms anywhere.

Grouped-query attention (GQA): keys and values may use fewer heads than queries
(kv_heads <= n_heads), and each kv head is shared by n_heads / kv_heads query
heads. With kv_heads = n_heads this is plain MHA.

Layout convention: activations are (batch, seq, dim); inside attention they are
(batch, heads, seq, head_dim).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_reference import attention, rope


def forward(
    x: NDArray[np.floating],
    weights: dict[str, NDArray[np.floating]],
    cos: NDArray[np.floating],
    sin: NDArray[np.floating],
    past_k: NDArray[np.floating] | None = None,
    past_v: NDArray[np.floating] | None = None,
    *,
    n_heads: int,
    kv_heads: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Run multi-head attention over x.

    Args:
        x: Input activations, shape (batch, seq, dim).
        weights: Projection matrices, no bias:
            'wq' (dim, n_heads * head_dim)
            'wk' (dim, kv_heads * head_dim)
            'wv' (dim, kv_heads * head_dim)
            'wo' (n_heads * head_dim, dim)
        cos: Full RoPE cos table, shape (max_seq, head_dim/2). Slice by past_len.
        sin: Full RoPE sin table, same shape as cos.
        past_k: Cached keys, shape (batch, n_heads, past_len, head_dim), or None.
        past_v: Cached values, same shape as past_k, or None.
        n_heads: Number of query heads.
        kv_heads: Number of key/value heads. Defaults to n_heads (plain MHA).

    Returns:
        out: Output activations, shape (batch, seq, dim).
        k_full, v_full: Updated cache, shape (batch, n_heads, past_len + seq, head_dim).
    """
    batch_dim = x.shape[0]
    seq_dim = x.shape[1]
    head_dim = weights["wq"].shape[-1] // n_heads
    if kv_heads is None:
        kv_heads = n_heads

    q = x @ weights["wq"]  # (batch, seq, n_heads * head_dim)
    k = x @ weights["wk"]  # (batch, seq, kv_heads * head_dim)
    v = x @ weights["wv"]  # (batch, seq, kv_heads * head_dim)

    q = q.reshape(batch_dim, seq_dim, n_heads, head_dim).transpose(
        0, 2, 1, 3
    )  # (batch, n_heads, seq, head_dim)
    k = k.reshape(batch_dim, seq_dim, kv_heads, head_dim).transpose(
        0, 2, 1, 3
    )  # (batch, kv_heads, seq, head_dim)
    v = v.reshape(batch_dim, seq_dim, kv_heads, head_dim).transpose(
        0, 2, 1, 3
    )  # (batch, kv_heads, seq, head_dim)

    past_len = 0 if past_k is None else past_k.shape[-2]

    cos_slice = cos[past_len : past_len + seq_dim]  # (seq, head_dim/2)
    sin_slice = sin[past_len : past_len + seq_dim]  # (seq, head_dim/2)
    q = rope.forward(q, cos_slice, sin_slice)
    k = rope.forward(k, cos_slice, sin_slice)

    if kv_heads < n_heads:
        group_size = n_heads // kv_heads
        k = np.repeat(k, group_size, axis=1)  # (batch, n_heads, seq, head_dim)
        v = np.repeat(v, group_size, axis=1)  # (batch, n_heads, seq, head_dim)

    out, k_full, v_full = attention.forward(q, k, v, past_k, past_v)
    out = out.transpose(0, 2, 1, 3).reshape(
        batch_dim, seq_dim, n_heads * head_dim
    )  # (batch, seq, n_heads * head_dim)
    out = out @ weights["wo"]  # (batch, seq, dim)
    return out, k_full, v_full
