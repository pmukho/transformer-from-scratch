"""Causal scaled dot-product attention, KV-cache-aware.

Given per-head queries, keys, and values:

    scores  = (q @ k.T) / sqrt(head_dim)
    scores  = scores masked so query at absolute position i sees only keys at
              positions j <= i (causal)
    weights = softmax(scores, axis=-1)
    out     = weights @ v

KV cache: on each step the new k/v are concatenated onto the cached past k/v
before the scores are computed, and the full k/v are returned to be stored back.
During decode q has length 1 while the cache holds all earlier positions.

RoPE is applied upstream in multihead.py (to q and to the new k before it is
cached), so q and k arrive here already rotated. Values are never rotated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def forward(
    q: NDArray[np.floating],
    k: NDArray[np.floating],
    v: NDArray[np.floating],
    past_k: NDArray[np.floating] | None = None,
    past_v: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Attend q over the concatenated (past + new) keys and values.

    Args:
        q: Queries for the current step, shape (..., new_len, head_dim). Leading
            dims are (batch and/or heads). In decode new_len is 1.
        k: Keys for the current step, shape (..., new_len, head_dim).
        v: Values for the current step, shape (..., new_len, head_dim).
        past_k: Cached keys from earlier steps, shape (..., past_len, head_dim),
            or None on the first (prefill) call.
        past_v: Cached values, shape (..., past_len, head_dim), or None.

    Returns:
        out: Attention output, shape (..., new_len, head_dim).
        k_full: past_k concatenated with k, shape (..., past_len + new_len, head_dim).
            Store this back into the cache.
        v_full: past_v concatenated with v, same shape as k_full.
    """
    k_full = np.concatenate([past_k, k], axis=-2) if past_k is not None else k
    v_full = np.concatenate([past_v, v], axis=-2) if past_v is not None else v

    head_dim = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k_full, -2, -1)) / np.sqrt(head_dim)

    new_len = q.shape[-2]
    total_len = k_full.shape[-2]
    causal_mask = np.tril(np.ones((new_len, total_len), dtype=bool), k=total_len - new_len)
    scores[..., ~causal_mask] = -np.inf

    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    out = np.matmul(weights, v_full)
    return out, k_full, v_full
