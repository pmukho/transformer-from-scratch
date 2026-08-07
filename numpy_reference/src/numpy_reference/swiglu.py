"""SwiGLU feed-forward block.

Gated feed-forward used in Llama. Three matrices, no bias:

    gate = x @ w_gate         (dim -> hidden)
    up   = x @ w_up           (dim -> hidden)
    out  = (silu(gate) * up) @ w_down    (hidden -> dim)

silu(z) = z * sigmoid(z). The gate path passes through silu and is multiplied
elementwise by the up path, then projected back to dim.

Pre-norm and the residual add live in block.py, so this is the pure FFN
transform: dim in, dim out.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def forward(
    x: NDArray[np.floating],
    weights: dict[str, NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Apply the SwiGLU feed-forward transform to x.

    Args:
        x: Input activations, shape (batch, seq, dim).
        weights: Projection matrices, no bias:
            'w_gate' (dim, hidden)
            'w_up'   (dim, hidden)
            'w_down' (hidden, dim)

    Returns:
        Output activations, shape (batch, seq, dim).
    """
    w_gate = weights["w_gate"]
    w_up = weights["w_up"]
    w_down = weights["w_down"]

    gate = x @ w_gate
    up = x @ w_up
    silu_gate = gate * (1 / (1 + np.exp(-gate)))
    out = (silu_gate * up) @ w_down
    return out
