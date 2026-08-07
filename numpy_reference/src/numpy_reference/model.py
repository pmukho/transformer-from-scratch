"""Full decoder stack.

Token embedding -> N decoder blocks -> final RMSNorm -> lm_head, where the
lm_head is weight-tied to the embedding (logits = h @ embedding.T).

Each block keeps its own KV cache, so the model's cache is a list with one
(k, v) entry per layer. All layers share the same absolute position, so they
all slice the RoPE tables by the same past_len (handled inside the block).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_reference import block, rmsnorm


def forward(
    tokens: NDArray[np.integer],
    weights: dict,
    cos: NDArray[np.floating],
    sin: NDArray[np.floating],
    past: list[tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None,
    *,
    n_heads: int,
    kv_heads: int | None = None,
) -> tuple[NDArray[np.floating], list[tuple[NDArray[np.floating], NDArray[np.floating]]]]:
    """Run the decoder stack and produce next-token logits.

    Args:
        tokens: Integer token ids, shape (batch, seq).
        weights: Dict:
            'embedding'  (vocab, dim)   token embedding, also the tied lm_head
            'blocks'     list           one per-block weight dict (see block.forward)
            'final_norm' (dim,)         RMSNorm gain before the lm_head
        cos, sin: RoPE tables, passed to every block.
        past: Per-layer KV cache as a list of (k, v), length n_layers, or None on
            prefill.
        n_heads, kv_heads: Head config, passed to every block.

    Returns:
        logits: Next-token scores, shape (batch, seq, vocab).
        new_past: Updated per-layer cache, list of (k_full, v_full), length n_layers.
    """
    x = weights["embedding"][tokens]  # (batch, seq, dim)

    new_past: list[tuple[NDArray[np.floating], NDArray[np.floating]]] = []
    for i, block_weights in enumerate(weights["blocks"]):
        past_k, past_v = (None, None) if past is None else past[i]
        x, k_full, v_full = block.forward(
            x,
            block_weights,
            cos,
            sin,
            past_k=past_k,
            past_v=past_v,
            n_heads=n_heads,
            kv_heads=kv_heads,
        )
        new_past.append((k_full, v_full))

    x = rmsnorm.forward(x, weights["final_norm"])

    logits = x @ weights["embedding"].T

    return logits, new_past
