"""Greedy autoregressive generation.

Drives the KV cache: prefill the prompt once, then generate one token at a time,
each step doing a length-1 forward against the cached context instead of
re-running the whole prefix. Greedy means the next token is always the argmax of
the last position's logits (no sampling).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_reference import model


def generate(
    tokens: NDArray[np.integer],
    weights: dict,
    cos: NDArray[np.floating],
    sin: NDArray[np.floating],
    max_new_tokens: int,
    *,
    n_heads: int,
    kv_heads: int | None = None,
) -> NDArray[np.integer]:
    """Greedily generate max_new_tokens continuation tokens.

    Args:
        tokens: Prompt token ids, shape (batch, prompt_len).
        weights: Model weights (see model.forward).
        cos, sin: RoPE tables. Must cover at least prompt_len + max_new_tokens
            positions, since generation walks the sequence forward.
        max_new_tokens: How many new tokens to generate.
        n_heads, kv_heads: Head config, passed to the model.

    Returns:
        Generated token ids, shape (batch, max_new_tokens).
    """
    batch, prompt_len = tokens.shape
    new_tokens: list[NDArray[np.integer]] = []

    # Prefill the prompt to get the initial past and the first next token.
    logits, past = model.forward(
        tokens,
        weights,
        cos,
        sin,
        past=None,
        n_heads=n_heads,
        kv_heads=kv_heads,
    )
    next_token = np.argmax(logits[:, -1, :], axis=-1)  # (batch,)
    new_tokens.append(next_token)

    # Generate one token at a time, feeding back the last token and the past.
    for _ in range(max_new_tokens - 1):
        logits, past = model.forward(
            next_token[:, None],  # (batch, 1)
            weights,
            cos,
            sin,
            past=past,
            n_heads=n_heads,
            kv_heads=kv_heads,
        )
        next_token = np.argmax(logits[:, -1, :], axis=-1)  # (batch,)
        new_tokens.append(next_token)

    return np.stack(new_tokens, axis=1)  # (batch, max_new_tokens)
